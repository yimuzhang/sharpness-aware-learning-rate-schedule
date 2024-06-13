# sharpness-aware-learning-rate-schedule
run: python src/sgd_mom.py cifar10-5k fc-tanh  mse  0.05 1000 --acc_goal 0.95 --neigs 1  --eig_freq 10 --batchsize 50 --beta 0.9   
学习率改变的频率和计算特征值的概率一致，可能需要取小一些  
beta是动量的系数
