import torch
import matplotlib.pyplot as plt
from os import environ
import matplotlib
#matplotlib.pyplot.show()

dataset = "cifar10-5k"
arch = "fc-tanh"
loss = "mse"
gd_lr = 0.05
gd_eig_freq = 1

gd_directory = f"E:\PEK_project\edge-of-stability-github\\res/{dataset}/{arch}/seed_1/{loss}/mom/lr_{gd_lr}"
gd_directory_1="E:\PEK_project\edge-of-stability-github\\baseline/cifar10-5k/fc-tanh/seed_1/mse/mom/lr_0.07"

gd_train_loss = torch.load(f"{gd_directory}/train_loss_final")
gd_train_acc = torch.load(f"{gd_directory}/train_acc_final")
gd_sharpness = torch.load(f"{gd_directory}/eigs_final")[:,0]
gd_lr = torch.load(f"{gd_directory}/lr_final")
gd_test= torch.load(f"{gd_directory}/test_loss_final")

gd_train_loss1 = torch.load(f"{gd_directory_1}/train_loss_final")
gd_train_acc1 = torch.load(f"{gd_directory_1}/train_acc_final")
gd_sharpness1 = torch.load(f"{gd_directory_1}/eigs_final")[:,0]
gd_test1= torch.load(f"{gd_directory_1}/test_loss_final")
gd_sig1=torch.load(f"{gd_directory_1}/sigma_final")
gd_lr1 = torch.load(f"{gd_directory_1}/lr_final")

gd_sig=torch.load(f"{gd_directory}/sigma_final")
print(gd_train_loss[2300:2400])
#gd_sig1=torch.load(f"{gd_directory_1}/sigma_final")
#print(min(gd_train_loss[5200:6950]))
#print(max(gd_sig[5200:6950]))
#print(gd_train_loss1[280:300])
print(gd_sig[0:2])
print(gd_sharpness[0:1])

plt.figure(figsize=(5, 5), dpi=100)

plt.subplot(4, 1, 1)
plt.plot(gd_train_loss[:],label="lr0=6*sig/sharpness,step=0")
plt.plot(gd_train_loss1[:4000],label="lr=6*sig/sharpness")
#plt.legend()
plt.ylim((0.1,0.45))
#matplotlib.pyplot.show()
plt.title("train loss")

plt.subplot(4, 1, 2)
plt.scatter(torch.arange(len(gd_lr)) * gd_eig_freq, gd_lr, s=5)
#plt.plot(gd_lr,label="lr0=6*sig/sharpness,step=0")
plt.plot(gd_lr1[0:4000],label="lr0=6*sig/sharpness,step=0")
#matplotlib.pyplot.show()
#plt.title("train accuracy")
#plt.plot(gd_sig1,label="lr=6*sig/sharpness")
#plt.legend()
plt.title("lr")

plt.subplot(4, 1, 3)
plt.scatter(torch.arange(len(gd_sharpness)) * 10, gd_sharpness, s=5)
plt.scatter(torch.arange(len(gd_sharpness1)) * 100, gd_sharpness1, s=5,c='y')
#plt.axhline(2. / gd_lr, linestyle='dotted')
#plt.plot(gd_sig*2/gd_lr,linestyle="dotted")
#plt.yticks(ticks=[50,60,70,80])
plt.xlim((0,4000))
plt.grid(True)
#matplotlib.pyplot.show()
plt.title("sharpness")
plt.xlabel("iteration")

plt.subplot(4, 1, 4)
plt.plot(gd_test[20:],label="lr0=6*sig/sharpness,step=0")
plt.plot(gd_test1[20:],label="lr0=6*sig/sharpness,step=0")
#matplotlib.pyplot.show()
#plt.title("train accuracy")
plt.ylim((0.35,0.6))
plt.xlim((0,4000))
#plt.plot(gd_sig1,label="lr=6*sig/sharpness")
#plt.legend()
plt.title("test")
matplotlib.pyplot.show()