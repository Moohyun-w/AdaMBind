import sys
import warnings
warnings.filterwarnings('ignore')
import random
import argparse  
import torch
from  torch import optim
import random
import pandas as pd
from utils.DataSplit import train_datasplit
from utils.TestbedDataset import TestbedDataset
from MetaTrain.MAML import MAML
import os
import numpy as np
from GNN_hw.gat import GATNet
from GNN_hw.gat_gcn import GAT_GCN
from GNN_hw.gcn import GCNNet
from GNN_hw.ginconv import GINConvNet
from utils.scheduler import Scheduler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


argparser = argparse.ArgumentParser()

argparser.add_argument('--root', type=str, default='C:/Users/l/Desktop/G1work/')
argparser.add_argument('--dataset', type=str, help='datasets', default='davis')
argparser.add_argument('--seed', type=int, help='random seed', default=210)
argparser.add_argument('--train_strategy', type=str, help='meta/single', default='meta')
argparser.add_argument('--mode', type=str, help='ratio/num', default='num')

#元学习模型设置
argparser.add_argument('--gnn', nargs='*', default=['gcn','gat','gin','gat_gcn'])
argparser.add_argument('--learning_setting', type=str, help='choosing the learning setting: few-shot, zero-shot and majority')
argparser.add_argument('--meta_lr', type=float, help='task-level outer update learning rate', default=1e-4)  
argparser.add_argument('--reg_lr', type=float, help='task-level inner update learning rate', default=1e-3)  
argparser.add_argument('--ntm_lr', type=float, help='task-level inner update learning rate', default=0.0005)
argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=3)  
argparser.add_argument('--update_step_train', type=int, help='update steps for metatraining(inner loop)', default=5)  
argparser.add_argument('--C', type=int, help='Number of basis', default=100)
argparser.add_argument('--R', type=int, help='Protein Index matrix vector length', default=100)
argparser.add_argument('--L', type=int, help='Protein embedding length', default=1000)
argparser.add_argument('--outer_iters', type=int, help='outer loop of maml', default=15) 
argparser.add_argument('--batch_size', type=int, help='batch size', default=8)  
argparser.add_argument('--spt', type=int, help='5/20/40/60', default=40)


argparser.add_argument('--noise', type=int, help='1/0', default=1)
argparser.add_argument('--adaptive_tasks', type=int, help='1/0', default=1)
argparser.add_argument('--buffer_size', type=int, default=15) 
argparser.add_argument('--meta_batch_size', type=int, default=8)  

#单任务模型设置
argparser.add_argument('--single_task_epoch', type=int, help='epoch of graphdta', default=150) 
argparser.add_argument('--st_batch_size', type=int, help='epoch of graphdta', default=512) 
argparser.add_argument('--st_lr', type=float, help='epoch of graphdta', default=0.0005) 

args = argparser.parse_args()

dataset_name={"kiba":"kiba-full-data","davis":"davis-full-data","bindingdb":"bindingdb-full-data"}
davis_csv=pd.read_csv(f'{args.root}/data/{dataset_name[args.dataset]}.csv')
davis=TestbedDataset(root=f'{args.root}/data', dataset=f'{dataset_name[args.dataset]}')
model_dict={'gcn':GCNNet,'gat':GATNet,'gat_gcn':GAT_GCN,'gin':GINConvNet}


def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
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

r=f'{args.root}/result/'


for j in args.gnn:  
    print(f"==============dataset:{args.dataset} ; drug encoder:{j} ; seed {args.seed} ; spt {args.spt}==============\n")
    seed_torch(args.seed)
    rng = np.random.RandomState(args.seed)
    args.reg_lr=1e-4
    args.meta_lr=1e-5
    maml_loss=[]
    maml_iters=args.outer_iters
    net=model_dict[j]().to(device)
    metalearner=MAML(net)
    min_loss=100
    best_epoch=0

    F_data=train_datasplit(args.mode,davis_csv,davis,args.spt,args.seed)   #60;100
    idx2=[]
    for d in F_data:
        if len(F_data[d][1])==0:
            idx2.append(d)
    for d in idx2:
        F_data.pop(d) 
    task_num=len(F_data)
    val_size=int(task_num*0.1)
    train_size=int(task_num*0.8)
    test_size=task_num-val_size-train_size
    protein=list(F_data.keys())
    random.shuffle(protein)
    train_val_idx=protein[:val_size+train_size]
    test_idx=protein[val_size+train_size:]

    if args.adaptive_tasks==1:
        scheduler = Scheduler(21, args.buffer_size, grad_indexes=list(range(21))).to(device)
        scheduler_optimizer = torch.optim.Adam(scheduler.parameters(), lr=0.0001)
    
    #=========================Meta learning module=============================
    if args.train_strategy=='meta':
        #注意修改
        # C:\Users\l\Desktop\G1work\Panpep\PanPep-main\result\bindingdb\seed42\ckpt
        if not os.path.exists(f"{r}/{args.dataset}/seed{str(args.seed)}/ckpt"):
            os.makedirs(f"{r}/{args.dataset}/seed{str(args.seed)}/ckpt")
        maml_path=f"{r}/{args.dataset}/seed{str(args.seed)}/ckpt/{str(args.spt)}_{j}.pt"
        if not os.path.exists(f'{r}/{args.dataset}/seed{str(args.seed)}/output'):
            os.makedirs(f'{r}/{args.dataset}/seed{str(args.seed)}/output')
        if not os.path.exists(f'{r}/{args.dataset}/seed{str(args.seed)}/log'):
            os.makedirs(f'{r}/{args.dataset}/seed{str(args.seed)}/log')
        #========================================================================================
        for it in range(maml_iters):
            sys.stdout.flush()  #
            if (it+1)%2==0:
                args.meta_lr*=0.8617
                args.reg_lr*=0.7617
                args.ntm_lr*=0.7617
            #=======================no adaptive=======================
            if args.adaptive_tasks==0:
                random.shuffle(train_val_idx)
                val_idx=train_val_idx[train_size:val_size+train_size]
                train_idx=train_val_idx[:train_size]
                metalearner.maml_train(net,args,it,train_idx,F_data)
                aff_pre,aff_ture,meta_loss,meta_ci,meta_r2,spear,pear=metalearner.maml_predict(net,args,val_idx,F_data)
                if meta_loss<min_loss:
                    min_loss=meta_loss
                    best_epoch=it+1
                    torch.save(metalearner.get_maml_init_params(), maml_path)
                    print('loss improved at epoch ', best_epoch, '; min_loss:',min_loss)

                else:
                    print('current_loss:',meta_loss,'current_r2:',meta_r2,';No improvement since epoch ', best_epoch, '; min_loss:',min_loss)

            #====================Adaptive task module=======================
            elif args.adaptive_tasks==1:
                #---------split meta train and meta val------------
                tr_task,val_task=meatloader(train_val_idx,args.buffer_size,args.meta_batch_size,nsplit=80)  #[[],[],[],...]
                count=0
                for train_idx,val_idx in zip(tr_task,val_task):

                    pt = int(count / (maml_iters * 219) * 100)
                    _ = metalearner.maml_train(net,args,it,train_idx,F_data,update=0)
                    tasks_loss=metalearner.get_mamltrain_loss()   #长度为15的损失列表
                    spt_grad,qry_grad=metalearner.get_grads()  #
                    task_losses, _, all_task_weight=scheduler.get_weight(tasks_loss,spt_grad,qry_grad,pt)
                    all_task_prob = torch.softmax(all_task_weight.reshape(-1), dim=-1)
                    selected_tasks_idx = scheduler.sample_task(all_task_prob, args.meta_batch_size)
                    selected_tasks= train_idx[selected_tasks_idx]

                    fast_weights=metalearner.maml_train(net,args,it,selected_tasks,F_data,update=0)

                    aff_pre,aff_ture,meta_loss,meta_ci,meta_r2,spear,pear=metalearner.maml_predict(net,args,val_idx,F_data,fast_weights)  
                    loss=0
                    for i in torch.tensor(selected_tasks_idx).to(device):
                        loss=loss-scheduler.m.log_prob(i)
                    loss*=(-meta_loss)
                    scheduler_optimizer.zero_grad()
                    loss.backward()
                    scheduler_optimizer.step()
                    meta_batch_loss, _, all_task_weight=scheduler.get_weight(tasks_loss,spt_grad,qry_grad,pt)
                    all_task_prob = torch.softmax(all_task_weight.reshape(-1), dim=-1)
                    selected_tasks_idx = scheduler.sample_task(all_task_prob, args.meta_batch_size)
                    selected_tasks_idx = torch.stack([torch.tensor(i) for i in selected_tasks_idx])
                    selected_tasks= train_idx[selected_tasks_idx]
                    metalearner.maml_train(net,args,it,selected_tasks,F_data,count=count,update=1)
                    count+=1
                
                    if count%10==0:
                        aff_pre,aff_ture,meta_loss,meta_ci,meta_r2,spear,pear=metalearner.maml_predict(net,args,val_idx,F_data)

                        if meta_loss<min_loss:
                            min_loss=meta_loss
                            best_epoch=it+1
                            torch.save(metalearner.get_maml_init_params(), maml_path)
                            print('loss improved at epoch ', best_epoch, '; current count:',count,'; min_loss:',min_loss)

                        else:
                            print('current_loss:',meta_loss,'current_r2:',meta_r2,';No improvement since epoch ', best_epoch, '; min_loss:',min_loss)
        
        y_pre_p=f"{r}/{args.dataset}/seed{str(args.seed)}/output/{str(args.spt)}_{j}_ypre.npy"
        y_real_p=f"{r}/{args.dataset}/seed{str(args.seed)}/output/{str(args.spt)}_{j}_ytrue.npy"
        net=model_dict[j]().to(device)
        try:
            maml_path=f"{r}/{args.dataset}/seed{str(args.seed)}/ckpt/{str(args.spt)}_{j}.pt"
            net.load_state_dict(torch.load(maml_path,map_location='cuda'))
        except Exception as e:
            maml_path=f"{r}/{args.dataset}/seed{str(args.seed)}/ckpt/meta_{str(args.spt)}_{j}.pt"
            net.load_state_dict(torch.load(maml_path,map_location='cuda'))
        metalearner=MAML(net)
        aff_pre,aff_ture,tes_meta_loss,tes_ci,tes_r2,spear,pear=metalearner.maml_predict(net,args,test_idx,F_data)
        aff_pre=np.array(aff_pre)
        np.save(y_pre_p,aff_pre)
        aff_ture=np.array(aff_ture)
        np.save(y_real_p,aff_ture)
        test_result_path=f"{r}/{args.dataset}/seed{str(args.seed)}/output/{str(args.spt)}_{j}.txt"

        f = open(test_result_path,'w')
        for i in [tes_meta_loss,tes_ci,tes_r2,spear,pear]:
            f.write(str(i)+'\n')
        f.close()
        print('final mse:',tes_meta_loss,';final ci: ', tes_ci, '; final r2:',tes_r2,'; final spearman:',spear,'; final pearson:',pear)
  