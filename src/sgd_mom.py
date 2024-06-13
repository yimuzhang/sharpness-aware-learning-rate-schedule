from os import makedirs

from torch.utils.data import RandomSampler, DataLoader
import torch
import numpy as np
from torch.nn.utils import parameters_to_vector
from torch.optim.lr_scheduler import CosineAnnealingLR

import argparse
import math

from archs import load_architecture
from utilities import get_gd_optimizer, get_gd_directory, get_loss_and_acc, compute_losses, \
    save_files, save_files_final, get_hessian_eigenvalues, iterate_dataset
from data import load_dataset, take_first, DATASETS
from torch.utils.data import TensorDataset
from torch.optim import SGD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
print(torch.__version__)




 
def compute_loss(network, loss_functions,sample,
                   batch_size):
    """Compute loss over a dataset."""
    L = len(loss_functions)
    losses = [0. for l in range(L)]
    preds=torch.zeros((batch_size,10))
    labs=torch.zeros((batch_size,10))
    cnt=0
    with torch.no_grad():
        for i in range(50):
            (x,y)=next(iter(sample))
            #(x,y)=dataset[i]
            
            x=x.to(device)
        #print(x)
            pred=network(x)
            #print(pred.size())
            labs=y
            preds=pred
            y=y.to(device)
            
            cnt=cnt+1
            for l ,loss_fn in enumerate(loss_functions):
                losses[l]+=loss_fn(pred,y) /batch_size
        for l ,lossfun in enumerate(loss_functions):
            losses[l]=losses[l]/50
    return losses,preds,labs



def main(dataset: str, arch_id: str, loss: str, opt: str, lr: float, max_steps: int, neigs: int = 0,
         physical_batch_size: int = 1000, eig_freq: int = -1, iterate_freq: int = -1, save_freq: int = -1,
         save_model: bool = False, beta: float = 0.0, nproj: int = 0,
         loss_goal: float = None, acc_goal: float = None, abridged_size: int = 5000, seed: int = 0,bacthsize: int=5000):
    seed=1
    #directory = get_gd_directory(dataset, lr, arch_id, seed, opt, loss, beta)
    res_dir="E:\PEK_project\edge-of-stability-github\\res"
    directory=f"{res_dir}/{dataset}/{arch_id}/seed_{seed}/{loss}/mom/lr_{lr}"
    if(opt=="sgd"):
        opt="gd"
    
    print(dataset,lr,arch_id,seed,opt,loss,beta)
    print(f"output directory: {directory}")
    makedirs(directory, exist_ok=True)

    train_dataset, test_dataset = load_dataset(dataset, loss)
    abridged_train = take_first(train_dataset, abridged_size)

    loss_fn, acc_fn = get_loss_and_acc(loss)

    torch.manual_seed(seed)
    network = load_architecture(arch_id, dataset).cuda()

    torch.manual_seed(7)
    projectors = torch.randn(nproj, len(parameters_to_vector(network.parameters())))

    optimizer = SGD(network.parameters(), lr=lr, momentum=0.9, nesterov=False)

    #optimizer = torch.optim.Adam(network.parameters(),0.0001)
    lr_change=eig_freq
    train_loss, test_loss, train_acc, test_acc = \
        torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps)
    iterates = torch.zeros(max_steps // iterate_freq if iterate_freq > 0 else 0, len(projectors))
    eigs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs)
    sig=torch.zeros(max_steps)
    lr_list = torch.zeros(max_steps)
    lr_list[0:4]=lr
    bs=bacthsize
    sharp=1
    lr_now=lr
    last_mom=0
    last_mom2=0
    sample=DataLoader(train_dataset,batch_size=bs,shuffle=True)

    for step in range(0, max_steps):
        sample0=DataLoader(train_dataset,batch_size=bs,shuffle=True)
        (x,y)=next(iter(sample))
        (train_loss[step], train_acc[step]),preds,labs = compute_loss(network, [loss_fn, acc_fn], sample0,bs)
                                                           
        test_loss[step], test_acc[step] = compute_losses(network, [loss_fn, acc_fn], test_dataset, physical_batch_size)

        if eig_freq != -1 and step % eig_freq == 0:
            eigs[step // eig_freq, :] = get_hessian_eigenvalues(network, loss_fn, abridged_train, neigs=neigs,
                                                                physical_batch_size=physical_batch_size)
            print("eigenvalues: ", eigs[step//eig_freq, :])
            sharp=eigs[step  // eig_freq ]

        if iterate_freq != -1 and step % iterate_freq == 0:
            iterates[step // iterate_freq, :] = projectors.mv(parameters_to_vector(network.parameters()).cpu().detach()) # mv means matrix * vector

        if save_freq != -1 and step % save_freq == 0:
            save_files(directory, [("eigs", eigs[:step // eig_freq]), ("iterates", iterates[:step // iterate_freq]),
                                   ("train_loss", train_loss[:step]), ("test_loss", test_loss[:step]),
                                   ("train_acc", train_acc[:step]), ("test_acc", test_acc[:step])])

        print(f"{step}\t{train_loss[step]:.3f}\t{train_acc[step]:.3f}\t{test_loss[step]:.3f}\t{test_acc[step]:.3f}")

        if (loss_goal != None and train_loss[step] < loss_goal) or (acc_goal != None and train_acc[step] > acc_goal):
            break
        e_grad=0
        e_grad2=0
        max_grad=0
        cnt=0
        
        #beta=0.9
        for i in range(100):# calculate E(grad) E(grad^2)
            sample0=DataLoader(train_dataset,batch_size=bs,shuffle=True)
            (x0,y0)=next(iter(sample0))
            optimizer.zero_grad()
            pred=network(x0.to(device))
            loss=loss_fn(pred,y0.to(device))/bs
            loss.backward()
            grad_in=torch.tensor([],device='cuda')
            for gra in network.parameters():
                grad_pa=gra.grad
                grad_pa=grad_pa.view(-1)
                grad_in=torch.cat((grad_in,grad_pa))
            e_grad=e_grad+grad_in
            e_grad2=e_grad2+torch.norm(grad_in,p=2)**2
            if torch.norm(grad_in,p=2)>max_grad:
                max_grad=torch.norm(grad_in,p=2)
            cnt=cnt+1
        sigma= torch.norm(e_grad/cnt,p=2)**2/max_grad.item()**2#torch.norm(e_grad2/cnt,p=1)-torch.norm(e_grad/cnt,p=2)**2
        sigma2=torch.norm(e_grad/cnt,p=2)**2/(e_grad2/cnt)


        optimizer.zero_grad()
        pred=network(x.to(device))
        loss=loss_fn(pred,y.to(device))/bs
        loss.backward()
        grad_in=torch.tensor([],device='cuda')
        for gra in network.parameters():
                grad_pa=gra.grad
                grad_pa=grad_pa.view(-1)
                grad_in=torch.cat((grad_in,grad_pa))
        print("grad norm:",torch.norm(grad_in,p=1))

        mom=last_mom*beta+(1-beta)*grad_in
        mom2=last_mom2*beta+(1-beta)*torch.norm(grad_in,p=2)**2 #?????not sure, need provement
        print("sigma:",sigma,sigma2,bs)
        print("momentum:",torch.norm(mom,p=2),mom2)
        print("E(grad):",torch.norm(e_grad/cnt,p=2),e_grad2/cnt)
        sig[step]=sigma2        
        if step%lr_change==0:#chang learning rate
            sharp=sharp.to(device)
            lr_now=(torch.norm(mom,p=2)**2+math.sqrt(torch.norm(mom,p=2)**4+0.01*train_loss[step]*sharp*mom2))/(sharp*mom2) #change lr, you can change mom to e_grad to see no momentum
           # lr_now=3*sigma/sharp
            lr=lr_now.item()
            print('lr:',lr,beta)
            optimizer=SGD(network.parameters(), lr, 0.9)
        last_mom=beta*last_mom+grad_in
        optimizer.step()
        lr_list[step]=optimizer.param_groups[0]['lr']

        
    



    save_files_final(directory,
                     [("eigs", eigs[:(step + 1) // eig_freq]), ("iterates", iterates[:(step + 1) // iterate_freq]),
                      ("train_loss", train_loss[:step + 1]), ("test_loss", test_loss[:step + 1]),
                      ("train_acc", train_acc[:step + 1]), ("test_acc", test_acc[:step + 1]),("sigma",sig[:step+1]),("lr",lr_list[:(step+1)])])
    if save_model:
        torch.save(network.state_dict(), f"{directory}/snapshot_final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train using gradient descent.")
    parser.add_argument("dataset", type=str, choices=DATASETS, help="which dataset to train",default="cifar10-5k")
    parser.add_argument("arch_id", type=str, help="which network architectures to train",default="fc-tanh")
    parser.add_argument("loss", type=str, choices=["ce", "mse"], help="which loss function to use",default="mse")
    parser.add_argument("lr", type=float, help="the learning rate",default=0.01)
    parser.add_argument("max_steps", type=int, help="the maximum number of gradient steps to train for",default=10000)
    parser.add_argument("--opt", type=str, choices=["gd", "polyak", "nesterov","sgd"],
                        help="which optimization algorithm to use", default="gd")
    parser.add_argument("--seed", type=int, help="the random seed used when initializing the network weights",
                        default=0)
    parser.add_argument("--beta", type=float, help="momentum parameter (used if opt = polyak or nesterov)")
    parser.add_argument("--physical_batch_size", type=int,
                        help="the maximum number of examples that we try to fit on the GPU at once", default=1000)
    parser.add_argument("--acc_goal", type=float,
                        help="terminate training if the train accuracy ever crosses this value",default=0.99)
    parser.add_argument("--loss_goal", type=float, help="terminate training if the train loss ever crosses this value",default=0.01)
    parser.add_argument("--neigs", type=int, help="the number of top eigenvalues to compute")
    parser.add_argument("--eig_freq", type=int, default=100,
                        help="the frequency at which we compute the top Hessian eigenvalues (-1 means never)")
    parser.add_argument("--nproj", type=int, default=0, help="the dimension of random projections")
    parser.add_argument("--iterate_freq", type=int, default=-1,
                        help="the frequency at which we save random projections of the iterates")
    parser.add_argument("--abridged_size", type=int, default=5000,
                        help="when computing top Hessian eigenvalues, use an abridged dataset of this size")
    parser.add_argument("--save_freq", type=int, default=-1,
                        help="the frequency at which we save resuls")
    parser.add_argument("--save_model", type=bool, default=False,
                        help="if 'true', save model weights at end of training")
    parser.add_argument("--batchsize",type=int,default=5000)
    args = parser.parse_args()

    main(dataset=args.dataset, arch_id=args.arch_id, loss=args.loss, opt=args.opt, lr=args.lr, max_steps=args.max_steps,
         neigs=args.neigs, physical_batch_size=args.physical_batch_size, eig_freq=args.eig_freq,
         iterate_freq=args.iterate_freq, save_freq=args.save_freq, save_model=args.save_model, beta=args.beta,
         nproj=args.nproj, loss_goal=args.loss_goal, acc_goal=args.acc_goal, abridged_size=args.abridged_size,
         seed=args.seed,bacthsize=args.batchsize)
