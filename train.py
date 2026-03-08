import sys
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import random
import argparse  
import torch
from  torch import optim
import random
import pandas as pd
import pickle
from utils.TestbedDataset import TestbedDataset
from model.Trainer import Trainer
from model.gat_gcn import GAT_GCN
from model.gcn import GCNNet
import json
from model.scheduler import Scheduler
from utils.DataSplit import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
argparser = argparse.ArgumentParser()
argparser.add_argument('--root', type=str, default='/mnt/data12t0/AdaMBind')
argparser.add_argument('--dataset', type=str, help='datasets', default='bindingdb')
argparser.add_argument('--seed', type=int, help='random seed', default=168)
argparser.add_argument('--train_strategy', type=str, help='meta/single', default='meta')
argparser.add_argument('--gnn', nargs='*', default=['gat_gcn'])
argparser.add_argument('--noise_val', type=float, default=0.2)
argparser.add_argument('--learning_setting', type=str, help='choosing the learning setting: few-shot, zero-shot and majority')
argparser.add_argument('--meta_lr', type=float, help='task-level outer update learning rate', default=1e-5) 
argparser.add_argument('--reg_lr', type=float, help='task-level inner update learning rate', default=1e-4)  
argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=3)  
argparser.add_argument('--update_step_train', type=int, help='update steps for metatraining(inner loop)', default=5)  
argparser.add_argument('--outer_iters', type=int, help='outer loop of maml', default=5)  
argparser.add_argument('--batch_size', type=int, help='batch size', default=8)  
argparser.add_argument('--nums', type=int, default=10)
argparser.add_argument('--noise', type=int, help='1/0', default=1)
argparser.add_argument('--adaptive_tasks', type=int, help='1/0', default=1)
argparser.add_argument('--buffer_size', type=int, default=15) 
argparser.add_argument('--meta_batch_size', type=int, default=8)  
argparser.add_argument('--ckpt_dir', type=str, default="/mnt/data12t0/AdaMBind/ckpt") 
argparser.add_argument('--result_dir', type=str, default="/mnt/data12t0/AdaMBind/result") 
args = argparser.parse_args()

dataset_name={"kiba":"kiba-full-data","davis":"davis-full-data","bindingdb":"bindingdb-full-data"}
davis_csv=pd.read_csv(f'{args.root}/data/{dataset_name[args.dataset]}.csv')
davis=TestbedDataset(root=f'{args.root}/data', dataset=f'{dataset_name[args.dataset]}')
model_dict={'gcn':GCNNet,'gat_gcn':GAT_GCN}

def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) 
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def meatloader(train_val_idx,n_buffers,meta_batch,nsplit=100):
    tr_task,val_task=[],[]
    for i in range(nsplit):  #
        select_idx_tr=rng.choice(train_val_idx,size=n_buffers,replace=False)
        random.shuffle(select_idx_tr)
        tr_task.append(select_idx_tr)
        select_idx_val=rng.choice(train_val_idx,size=meta_batch,replace=False)
        random.shuffle(select_idx_val)
        val_task.append(select_idx_val)
    return tr_task,val_task

def read_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"file does not exist:{path}")
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

for j in args.gnn:  
    print(f"==============dataset:{args.dataset} ; drug encoder:{j} ; seed {args.seed} ; noise {args.noise_val}==============\n")
    seed_torch(args.seed)
    rng = np.random.RandomState(args.seed)
    maml_iters=args.outer_iters
    net=model_dict[j]().to(device)
    trainer=Trainer(net)
    min_loss=100
    best_epoch=0
    F_data=train_datasplit(davis_csv,davis,args.nums)
    idx2=[]
    for d in F_data:
        if len(F_data[d][1])==0:
            idx2.append(d)
    for d in idx2:
        F_data.pop(d) 
    if os.path.exists(f'{args.root}/data/{args.dataset}_1.txt'):
        train_idxs=read_file(f'{args.root}/data/{args.dataset}_1.txt')
        val_idxs=read_file(f'{args.root}/data/{args.dataset}_2.txt')
        test_idxs=read_file(f'{args.root}/data/{args.dataset}_3.txt')
    else:
        keys = list(F_data.keys())
        random.shuffle(keys)
        total = len(keys)
        train_size = int(total * 0.8) 
        val_size = int(total * 0.1)    
        train_idxs = keys[:train_size]
        val_idxs = keys[train_size:train_size + val_size]
        test_idxs = keys[train_size + val_size:]
    scheduler = Scheduler(21, args.buffer_size, list(range(21))).to(device)
    scheduler_optimizer = torch.optim.Adam(scheduler.parameters(), lr=0.0001)
    if not os.path.exists(f'{args.result_dir}/seed{str(args.seed)}/'):
        os.makedirs(f'{args.result_dir}/seed{str(args.seed)}/')
    for it in range(maml_iters):
        sys.stdout.flush() 
        if (it+1)%2==0:
            args.meta_lr*=0.8617
            args.reg_lr*=0.7617
        train_val_idx=train_idxs+val_idxs
        tr_task,val_task=meatloader(train_val_idx,args.buffer_size,args.meta_batch_size,nsplit=100) 
        count=0
        for train_idx,val_idx in zip(tr_task,val_task):
            pt = int(count / (maml_iters * 219) * 100)
            _ = trainer.train(net,args,it,train_idx,F_data,update=0)
            all_loss=trainer.get_rloss()   
            grad1,grad2=trainer.get_grads() 
            _, _, scores=scheduler.ats(all_loss,grad1,grad2,pt)
            all_task_prob = torch.softmax(scores.reshape(-1), dim=-1)
            sidx = scheduler.sample_task(all_task_prob, args.meta_batch_size)
            stasks= train_idx[sidx]
            fast_weights=trainer.train(net,args,it,stasks,F_data,update=0)
            aff_pre,aff_ture,loss_m,meta_ci,meta_r2,spear,pear=trainer.predict(net,args,val_idx,F_data,fast_weights)  
            loss=0
            for i in torch.tensor(sidx).to(device):
                loss=loss-scheduler.m.log_prob(i)
            loss*=(-loss_m)
            scheduler_optimizer.zero_grad()
            loss.backward()
            scheduler_optimizer.step()
            _, _, scores=scheduler.ats(all_loss,grad1,grad2,pt)
            all_task_prob = torch.softmax(scores.reshape(-1), dim=-1)
            sidx = scheduler.sample_task(all_task_prob, args.meta_batch_size)
            sidx = torch.stack([torch.tensor(i) for i in sidx])
            stasks= train_idx[sidx]
            trainer.train(net,args,it,stasks,F_data,count=count,update=1)
            model_save_path=f'{args.ckpt_dir}/{args.nums}_{args.seed}_{args.dataset}.pt'
            if count%10==0:
                aff_pre,aff_ture,loss_m,meta_ci,meta_r2,spear,pear=trainer.predict(net,args,val_idx,F_data)
                if loss_m<min_loss:
                    min_loss=loss_m
                    best_epoch=it+1
                    torch.save(trainer.get_params(), model_save_path)
                    print('loss improved at epoch ', best_epoch, '; current count:',count,'; min_loss:',min_loss)
                else:
                    print('current_loss:',loss_m,'current_r2:',meta_r2,';No improvement since epoch ', best_epoch, '; min_loss:',min_loss)
            count+=1
    net=model_dict[j]().to(device)
    net.load_state_dict(torch.load(model_save_path,map_location='cuda'))
    trainer=Trainer(net)
    aff_pre,aff_ture,tes_loss,tes_ci,tes_r2,spear,pear=trainer.predict(net,args,test_idxs,F_data)
    test_result_path=f"{args.result_dir}/{args.dataset}_{args.nums}_{args.seed}.txt"
    f = open(test_result_path,'w')
    for i in [tes_loss,tes_ci,tes_r2,spear,pear]:
        f.write(str(i)+'\n')
    f.close()
    print('final mse:',tes_loss,';final ci: ', tes_ci, '; final r2:',tes_r2,'; final spearman:',spear,'; final pearson:',pear)