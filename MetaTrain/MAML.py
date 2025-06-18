import torch
from  torch import optim
from torch.nn import functional as F
import torch.nn as nn
from random import shuffle
from utils.TestbedDataset import *
from utils.criterion import *
# from GNNs.gat import GATNet
from GNN_hw.gat import GATNet
# from GNNs.gat_gcn import GAT_GCN
from GNN_hw.gat_gcn import GAT_GCN
# from GNNs.gcn import GCNNet
from GNN_hw.gcn import GCNNet
# from GNNs.ginconv import GINConvNet
from GNN_hw.ginconv import GINConvNet
import torch.nn.utils as utils
import copy
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dict={'gcn':GCNNet,'gat':GATNet,'gat_gcn':GAT_GCN,'gin':GINConvNet}
def _split_parameters(x,memory_parameters):
    """
    This function is used to rebuild the model parameter shape from the parameter vector
    
    Parameters:
        param x: parameter vector
        param memory_parameters: origin model parameter shape
        
    Returns:
        a new model parameter shape from the parameter vector
    """
    
    new_weights = {}
    start_index = 0
    names=list(memory_parameters.keys())
    for i in names:
        end_index = np.prod(memory_parameters[i].shape)
        memory_parameters[i]=x[:,start_index:start_index+end_index].reshape(memory_parameters[i].shape)
        start_index += end_index
    return memory_parameters
    

class MAML():
    def __init__(self,net):
        self.maml_init_params=copy.deepcopy(net.state_dict())
        # self.maml_init_params=copy.deepcopy(net.parameters())
        self.maml_gradients=[]
        self.maml_gradients_spt=[]
        self.maml_train_loss=[]
        self.val_r2=[]
        self.predict_dict={}
        # self.decay_rate=0.97
        for param in net.parameters():
            grad = torch.zeros_like(param.data)
            self.maml_gradients.append(grad)
            self.maml_gradients_spt.append(grad)

    def zero_grad(self):
        for i in range(len(self.maml_gradients)):
            self.maml_gradients[i].zero_()
  
    def maml_train(self,net,args,epoch,train_idx,F_data,count=None,update=1):
        print('Train model on {} training tasks...'.format(len(train_idx)))
        self.model_params=[]
        task_num = len(train_idx)
        self.loss_list_qry = [0]*task_num
        optimizer= torch.optim.Adam(net.parameters(), lr=args.reg_lr)
        loss_fn=nn.MSELoss()
        self.grads_spt_list = []
        self.grads_qry_list=[]
        #------1.内循环训练子任务-----
        #在各个训练任务上依次训练回归器
        for n,i in enumerate(train_idx):  #取第i个训练任务

            net.load_state_dict(self.maml_init_params) #第i个任务的初始模型
            params_vector = utils.parameters_to_vector(net.parameters()) #将maml的模型参数平铺为一个向量（不能在maml更新时再保存params_vector，因为maml更新时net.parameters()的取值已改变！）

            # print('meta Train epoch: {} \t train model on the {}th/{} task'.format(epoch+1,n+1,task_num))
            qry_loss=0.  #初始化查询集损失
            total_preds = torch.Tensor() #模型在第i个任务的查询集上的预测值
            total_labels = torch.Tensor() #模型在第i个任务的查询集的真实值
            spt_loader=DataLoader(F_data[i][0],batch_size=args.batch_size,shuffle=True) #mini batch
            qry_loader=DataLoader(F_data[i][1],batch_size=args.batch_size,shuffle=False)
            for iter in range(args.update_step_train): #训练args.update_step_train次
                for batch_idx,data in enumerate(spt_loader):
                    data=data.to(device)
                    #reset
                    optimizer.zero_grad()
                    #forward
                    output = net(data) # (ways * shots, ways)
                    #loss
                    data_num=len(data)
                    if args.noise==1:
                        noise=torch.tensor(np.random.uniform(-0.5,0.5,data_num)).to(device)
                    # y=data.y+noise
                    if data.y.ndim==1:
                        if args.noise==1:
                            y=data.y+noise
                        loss = loss_fn(output, y.view(-1, 1).float().to(device))
                    elif data.y.ndim==2:
                        if args.noise==1:
                            y=data.y[:,1]+noise
                        loss = loss_fn(output, data.y[:,1].view(-1, 1).float().to(device))
                    #backward
                    loss.backward()
                    #update
                    optimizer.step()
            #-----支撑集上梯度------
            grads = []
            for param in net.parameters():
                grad = param.grad
                grads.append(grad)
            self.grads_spt_list.append(grads)

            #查询集测试
            # with torch.no_grad():
            for data in qry_loader:
                data=data.to(device)
                optimizer.zero_grad()
                output=net(data)
                #loss
                data_num=len(data)
                if args.noise==1:
                    noise=torch.tensor(np.random.uniform(-0.5,0.5,data_num)).to(device)

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

            qry_loss=torch.tensor(mse(total_labels,total_preds))  #第i个任务在查询集上的损失
            self.loss_list_qry[n]=qry_loss
            params = [param.detach().cpu().flatten() for param in net.parameters()]  #第i个任务模型参数
            self.model_params.append(torch.cat(params, dim=0))  

            #取每个任务的回归器的梯度
            grads = []
            for param in net.parameters():
                grad = param.grad
                grads.append(grad)
            self.grads_qry_list.append(grads)
            # #将每个任务梯度加入元学习器梯度中
            for k in range(len(self.maml_gradients)):
                if grads[k] is None:
                    pass 
                else:
                    self.maml_gradients[k]+=grads[k]  
                    # grad_index.append(k)
                      
        #-------------------------内循环训练结束--------------------------
        
        #-------外循环MAML更新参数-------  
        for k in range(len(self.maml_gradients)):
                if grads[k] is None:
                    pass 
                else:
                    self.maml_gradients[k]= self.maml_gradients[k]/task_num
        #loss
        meta_loss= sum(self.loss_list_qry) / task_num  #元学习器损失
        
        #---update----
        # 将梯度展平为一个向量
        # for k in range(len(self.maml_gradients)):
        #         self.maml_gradients[k]/=task_num 
        gradients_vector = utils.parameters_to_vector(self.maml_gradients)
        # 更新模型参数向量
        params_vector -= args.meta_lr * gradients_vector
        # 将更新后的参数向量还原为模型参数
        utils.vector_to_parameters(params_vector, net.parameters())
        #reset
        # self.zero_grad()
        #保存元学习器初始参数
        if update==1:
            self.maml_init_params=copy.deepcopy(net.state_dict())
            if count%10==0:
                print('Train epoch: {} \tmeta Loss: {:.6f}'.format(epoch+1,
                                                            meta_loss.cpu().detach()))
        else:
            fast_weights=net.state_dict()
            return fast_weights
            
    def maml_predict(self,net,args,val_idx,F_data,fast_weights=None,ct=True):
        print('Make prediction on {} validation tasks...'.format(len(val_idx)))
        self.model_params=[]
        task_num = len(val_idx)
        loss_list_qry = [0]*task_num
        loss_fn=nn.MSELoss()
        optimizer= torch.optim.Adam(net.parameters(), lr=args.reg_lr)
        # preds,labels=np.array(),np.array()
        #------1.内循环训练子任务-----
        #在各个训练任务上依次训练回归器
        if fast_weights!=None:
            self.maml_init_params=fast_weights
        preds,labels=[],[]
        for n,i in enumerate(val_idx):  #取第i个训练任务
            # print('finetune model on the {}th task'.format(n+1))
            net.load_state_dict(self.maml_init_params) #第i个任务的初始模型
            qry_loss=0.  #初始化查询集损失
            total_preds = torch.Tensor() #模型在第i个任务的查询集上的预测值
            total_labels = torch.Tensor() #模型在第i个任务的查询集的真实值
            qry_loader=DataLoader(F_data[i][1],batch_size=2,shuffle=False)
            try:
                spt_loader=DataLoader(F_data[i][0][:args.spt],batch_size=2,shuffle=True) #mini batch
                for iter in range(args.update_step_test): #训练args.update_step_test次
                    for batch_idx,data in enumerate(spt_loader):
                        data=data.to(device)
                        #reset
                        optimizer.zero_grad()
                        #forward
                        output = net(data) # (ways * shots, ways)
                        if data.y.ndim==1:
                            loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
                        elif data.y.ndim==2:
                            loss = loss_fn(output, data.y[:,1].view(-1, 1).float().to(device))
                        #backward
                        loss.backward()
                        #update
                        optimizer.step()
            except Exception as e:
                pass

            #查询集测试
            with torch.no_grad():
                for data in qry_loader:
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
            self.val_r2.append(rsquare(total_labels,total_preds))
            self.predict_dict[i]=[total_labels,total_preds]
            params = [param.detach().cpu().flatten() for param in net.parameters()]  #第i个任务模型参数
            self.model_params.append(torch.cat(params, dim=0))
            print(f'task {n+1} is done')
        labels,preds=np.array(labels),np.array(preds)
        # self.maml_train_loss.append(qry_loss)
        #-------计算元学习器损失-------
        if ct:
            val_loss=mse(labels,preds)
            val_ci=ci(labels,preds)
            val_r2=rsquare(labels,preds)
            val_spear=spearman(labels,preds)
            val_pear=pearson(labels,preds)
            
            return preds,labels,val_loss,val_ci,val_r2,val_spear,val_pear
        else:
            return preds,labels

    def get_mamltrain_loss(self):
        return self.loss_list_qry

    def get_params_matrix(self):
        params_matrix=torch.stack(self.model_params,dim=0)
        return params_matrix

    def get_maml_init_params(self):
        return self.maml_init_params
    
    def get_grads(self):
        return self.grads_spt_list,self.grads_qry_list
    
    def get_rscores(self):
        return self.val_r2
    
    def get_single_tsk_res(self,tsk):
        return self.predict_dict[tsk]