import torch

from cnn import ConvNet

from Dataloader import get_train_data, test_val_data

from train_test import train_model,test_model

""" 超参数"""
batch_size = 64
epoch = 10
learning_rate = 0.001
momentum = 0.9

cnn_model = ConvNet().cuda()

loss_func = torch.nn.CrossEntropyLoss()

sgd_opt = torch.optim.SGD(cnn_model.parameters(),
                          lr = learning_rate,
                          momentum= momentum)

# 读取数据
_, train_loader = get_train_data()
_,test_loader,val_loader = test_val_data()

# 开始训练
train_model(cnn_model, train_loader, val_loader, batch_size, epoch, loss_func, sgd_opt)
test_model(cnn_model,test_loader)

torch.save(cnn_model.state.dict(),'./cnnmodel')