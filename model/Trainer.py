import torch
from  torch import optim
from torch.nn import functional as F
import torch.nn as nn
from random import shuffle
from utils.TestbedDataset import *
from utils.criterion import *
import torch.nn.utils as utils
import copy
import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def _split_parameters(x,memory_parameters):
    new_weights = {}
    start_index = 0
    names=list(memory_parameters.keys())
    for i in names:
        end_index = np.prod(memory_parameters[i].shape)
        memory_parameters[i]=x[:,start_index:start_index+end_index].reshape(memory_parameters[i].shape)
        start_index += end_index
    return memory_parameters

class Trainer():
    def __init__(self,net):
        self.init_params=copy.deepcopy(net.state_dict())
        self.gradients=[]
        for param in net.parameters():
            grad = torch.zeros_like(param.data)
            self.gradients.append(grad)

    def zero_grad(self):
        for i in range(len(self.gradients)):
            self.gradients[i].zero_()
  
    def train(self,net,args,epoch,train_idx,F_data,count=None,update=1):
        print('Train model on {} training tasks...'.format(len(train_idx)))
        self.model_params=[]
        task_num = len(train_idx)
        self.loss_qt = [0]*task_num
        self.loss_r=[0]*task_num
        optimizer= torch.optim.Adam(net.parameters(), lr=args.reg_lr)
        loss_fn=nn.MSELoss()
        self.grads1 = []
        self.grads2=[]
        for n,i in enumerate(train_idx): 
            net.load_state_dict(self.init_params) 
            params_vector = utils.parameters_to_vector(net.parameters()) 
            loss_t=0. 
            total_preds = torch.Tensor() 
            total_labels = torch.Tensor() 
            loader=DataLoader(F_data[i][0],batch_size=args.batch_size,shuffle=True) 
            for iter in range(args.update_step_train): 
                for batch_idx,data in enumerate(loader):
                    data=data.to(device)
                    optimizer.zero_grad()
                    output = net(data) 
                    data_num=len(data)
                    if args.noise==1:
                        noise=torch.tensor(np.random.uniform(-args.noise_val,args.noise_val,data_num)).to(device)
                    if data.y.ndim==1:
                        if args.noise==1:
                            y=data.y+noise
                        loss = loss_fn(output, y.view(-1, 1).float().to(device))
                    elif data.y.ndim==2:
                        if args.noise==1:
                            y=data.y[:,1]+noise
                        loss = loss_fn(output, data.y[:,1].view(-1, 1).float().to(device))
                    loss.backward()
                    optimizer.step()
            grads = []
            for param in net.parameters():
                grad = param.grad
                grads.append(grad)
            self.grads1.append(grads)
            loader=DataLoader(F_data[i][1],batch_size=args.batch_size,shuffle=False)
            for data in loader:
                data=data.to(device)
                output=net(data)
                data_num=len(data)
                if args.noise==1:
                    noise=torch.tensor(np.random.uniform(-args.noise_val,args.noise_val,data_num)).to(device)
                if data.y.ndim==1:
                    if args.noise==1:
                        y=data.y+noise
                elif data.y.ndim==2:
                    if args.noise==1:
                        y=data.y[:,1]+noise
                total_preds = torch.cat((total_preds, output.cpu().detach()), 0)
                if data.y.ndim==1:
                    total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
                elif data.y.ndim==2:
                    total_labels = torch.cat((total_labels, data.y[:,1].view(-1, 1).cpu()), 0)
            total_labels,total_preds=total_labels.numpy().flatten(),total_preds.numpy().flatten()
            loss_t=torch.tensor(mse(total_labels,total_preds)) 
            self.loss_qt[n]=loss_t
            grads,r_loss = self.single(net,loader,optimizer,args)
            self.grads2.append(grads)
            self.loss_r[n]=r_loss
            for k in range(len(self.gradients)):
                if grads[k] is None:
                    pass 
                else:
                    self.gradients[k]+=grads[k]  
        for k in range(len(self.gradients)):
                if grads[k] is None:
                    pass 
                else:
                    self.gradients[k]= self.gradients[k]/task_num
        c_loss= sum(self.loss_qt) / task_num  
        gradients_vector = utils.parameters_to_vector(self.gradients)
        params_vector -= args.meta_lr * gradients_vector
        utils.vector_to_parameters(params_vector, net.parameters())
        if update==1:
            self.init_params=copy.deepcopy(net.state_dict())
            if count%10==0:
                print('Train epoch: {} \tCurrent Loss: {:.6f}'.format(epoch+1,
                                           c_loss.cpu().detach()))
        else:
            fast_weights=net.state_dict()
            return fast_weights

    def predict(self,net,args,val_idx,F_data,fast_weights=None,ct=True):
        print('Make prediction on {} validation tasks...'.format(len(val_idx)))
        self.model_params=[]
        task_num = len(val_idx)
        loss_fn=nn.MSELoss()
        optimizer= torch.optim.Adam(net.parameters(), lr=args.reg_lr)
        if fast_weights!=None:
            self.init_params=fast_weights
        preds,labels=[],[]
        for n,i in enumerate(val_idx): 
            net.load_state_dict(self.init_params) 
            total_preds = torch.Tensor()
            total_labels = torch.Tensor()
            loader=DataLoader(F_data[i][0],batch_size=2,shuffle=True)
            for iter in range(args.update_step_test): 
                for batch_idx,data in enumerate(loader):
                    data=data.to(device)
                    optimizer.zero_grad()
                    output = net(data)
                    if data.y.ndim==1:
                        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
                    elif data.y.ndim==2:
                        loss = loss_fn(output, data.y[:,1].view(-1, 1).float().to(device))
                    loss.backward()
                    optimizer.step()
            loader=DataLoader(F_data[i][1],batch_size=2,shuffle=False)
            with torch.no_grad():
                for data in loader:
                    data=data.to(device)
                    output=net(data)
                    total_preds = torch.cat((total_preds, output.cpu()), 0)
                    if data.y.ndim==1:
                        total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
                    elif data.y.ndim==2:
                        total_labels = torch.cat((total_labels, data.y[:,0].view(-1, 1).cpu()), 0)
                total_labels,total_preds=list(total_labels.numpy().flatten()),list(total_preds.numpy().flatten())
            labels+=total_labels
            preds+=total_preds
            print(f'task {n+1} is done')
        labels,preds=np.array(labels),np.array(preds)

        if ct:
            val_loss=mse(labels,preds)
            val_ci=ci(labels,preds)
            val_r2=rsquare(labels,preds)
            val_spear=spearman(labels,preds)
            val_pear=pearson(labels,preds)
            
            return preds,labels,val_loss,val_ci,val_r2,val_spear,val_pear
        else:
            return preds,labels

    def single(self,net,loader,optimizer,args):
        loss_fn=nn.MSELoss()
        total_preds = torch.Tensor() 
        total_labels = torch.Tensor() 
        for data in loader:
            data=data.to(device)
            optimizer.zero_grad()
            output=net(data)
            data_num=len(data)
            if args.noise==1:
                noise=torch.tensor(np.random.uniform(-args.noise_val,args.noise_val,data_num)).to(device)
            if data.y.ndim==1:
                if args.noise==1:
                    y=data.y+noise
                loss = loss_fn(output, y.view(-1, 1).float().to(device))
            elif data.y.ndim==2:
                if args.noise==1:
                    y=data.y[:,1]+noise
                loss = loss_fn(output, data.y[:,1].view(-1, 1).float().to(device))
            loss.backward()
            optimizer.step()
            total_preds = torch.cat((total_preds, output.cpu().detach()), 0)
            if data.y.ndim==1:
                total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
            elif data.y.ndim==2:
                total_labels = torch.cat((total_labels, data.y[:,1].view(-1, 1).cpu()), 0)
        total_labels,total_preds=total_labels.numpy().flatten(),total_preds.numpy().flatten()
        loss_r=torch.tensor(mse(total_labels,total_preds)) 
        grads = []
        for param in net.parameters():
            grad = param.grad
            grads.append(grad)
        return grads,loss_r

    def get_rloss(self):
        return self.loss_r

    def get_params(self):
        return self.init_params
    
    def get_grads(self):
        return self.grads1,self.grads2
    
