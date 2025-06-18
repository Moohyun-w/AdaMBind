import  torch
import math
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch_geometric.nn import global_max_pool as gmp,global_add_pool as gap
from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.utils import add_self_loops
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_sparse import SparseTensor, matmul
from torch import Tensor
from torch_scatter import scatter_add
# from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

config=[
    ('conv1',[[32,78],[32,32]]),
    ('bn1',[32]),
    ('conv2',[[32,32],[32,32]]),
    ('bn2',[32]), 
    ('conv3',[[32,32],[32,32]]),
    ('bn3',[32]), 
    ('conv4',[[32,32],[32,32]]),
    ('bn4',[32]),
    ('conv5',[[32,32],[32,32]]),
    ('bn5',[32]),
    ('gap',[True]),
    ('linear1', [128,32]),

    ('embedding', [26,128]),
    ('conv1d',[32,1000,8]),
    ('linear2', [128,32*121,]),

    # ('attention',[128,128]),
    ('fc1',[1024,256]),
    ('fc2',[256,1024]),
    ('fc3',[1,256]),

    ('dropout', [0.])
]

class GINConvNet(MessagePassing):
    def __init__(self,**kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GINConvNet, self).__init__(**kwargs)

        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()
        #GNN层中线性层初始化：weights->kaiming_normal_   ;  bias->zeros
        #普通线性层初始化：weights->kaiming_uniform_   ; bias->uniform_

        for i, (name, param) in enumerate(config):  
            # define a conv2d layer
            if name in ['conv1','conv2','conv3','conv4','conv5']:
                for pa in param:
                    w = nn.Parameter(torch.ones(*pa))
                    b = nn.Parameter(torch.zeros(pa[0]))
                    torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))
                    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(w)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    torch.nn.init.uniform_(b, -bound, bound)
                    self.vars.append(w)
                    self.vars.append(b)
                    # w = nn.Parameter(torch.ones(*pa))
                    # torch.nn.init.xavier_normal_(w)
                    # self.vars.append(w)
                    # self.vars.append(nn.Parameter(torch.zeros(pa[0])))  #0,1

            elif name in ['bn1','bn2','bn3','bn4','bn5']:
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])  #2,3

            elif name is 'attention':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))  #15,16

            elif name in ['linear1','linear2','fc1','fc2','fc3']:
                w = nn.Parameter(torch.ones(*param))
                b = nn.Parameter(torch.zeros(param[0]))
                torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(w)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.uniform_(b, -bound, bound)
                self.vars.append(w)
                self.vars.append(b)
                # w = nn.Parameter(torch.ones(*param))
                # torch.nn.init.xavier_normal_(w)
                # self.vars.append(w)
                # self.vars.append(nn.Parameter(torch.zeros(param[0])))   #8,9

            elif name is 'embedding':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.normal_(w)
                self.vars.append(w)  #10

            elif name is 'conv1d':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))  #11,12

            # elif name is 'linear2':
            #     w = nn.Parameter(torch.ones(*param))
            #     torch.nn.init.kaiming_normal_(w)
            #     self.vars.append(w)
            #     self.vars.append(nn.Parameter(torch.zeros(param[0])))  #13,14

            # elif name is 'fc1':
            #     w = nn.Parameter(torch.ones(*param))
            #     torch.nn.init.kaiming_normal_(w)
            #     self.vars.append(w)
            #     self.vars.append(nn.Parameter(torch.zeros(param[0])))   #15,16
            
            # elif name is 'fc2':
            #     w = nn.Parameter(torch.ones(*param))
            #     torch.nn.init.kaiming_normal_(w)
            #     self.vars.append(w)
            #     self.vars.append(nn.Parameter(torch.zeros(param[0])))  #17,18

            # elif name is 'fc3':
            #     w = nn.Parameter(torch.ones(*param))
            #     torch.nn.init.kaiming_normal_(w)
            #     self.vars.append(w)
            #     self.vars.append(nn.Parameter(torch.zeros(param[0])))   #19,20

            elif name is 'dropout':
                self.dropout = nn.Dropout(*param)

            elif name in ['gap','relu']:
                continue
            
        self.initial_eps=0.
        self.register_buffer('eps', torch.Tensor([0.]))
        self.eps.data.fill_(self.initial_eps)

    def forward(self, data,vars=None,bn_training=False):
        # get graph input
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # get protein input
        target = data.target

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx=0

        for name, param in config:
            #------------------获得药物嵌入-----------------
            if name in ['conv1','conv2','conv3','conv4','conv5']:
                if isinstance(x, Tensor):
                    x: OptPairTensor = (x, x)
                # propagate_type: (x: OptPairTensor)
                out = self.propagate(edge_index, x=x, size=None)
                x_r = x[1]
                if x_r is not None:
                    out += (1 + self.eps) * x_r
                w1,b1=vars[idx],vars[idx+1]
                w2,b2=vars[idx+2],vars[idx+3]
                x=F.linear(out,w1,b1)
                x=F.relu(x)
                x=F.linear(x,w2,b2)
                x=F.relu(x)
                idx += 4  #0~5
            
            elif name in ['bn1','bn2','bn3','bn4','bn5']:
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2 #6~9
            
            elif name is 'gap':
                x=gap(x,batch)

            elif name is 'linear1':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                x=F.relu(x) 
                x=F.dropout(x,p=0.2,training=self.training)
                idx += 2 
            #---------------------------------------

            #-------------获得靶蛋白嵌入-------------
            elif name is 'embedding':
                w= vars[idx]
                embedded_xt = F.embedding(target,w)
                idx+=1  #10
            
            elif name is 'conv1d':
                w, b = vars[idx], vars[idx + 1] 
                xt=F.conv1d(embedded_xt,w,b).view(-1, 32 * 121)
                idx+=2  #11,12
            
            elif name is 'linear2':
                w, b = vars[idx], vars[idx + 1]
                xt = F.linear(xt, w, b)
                xc = torch.cat((x, xt), 1)
                idx += 2  #13,14
            
            # -------------------注意力--------------------
            # elif name is 'attention':
            #     w,b=vars[idx], vars[idx + 1]
            #     att_d = F.relu(torch.einsum('ijk,kp->ijp', x.unsqueeze(2), w) + b)
            #     att_t = F.relu(torch.einsum('ijk,kp->ijp', xt.unsqueeze(2), w) +b)
            #     alph = torch.tanh(torch.einsum('aji,aik->ajk', att_d, att_t.transpose(1, 2)))
            #     alphdrug = torch.tanh(alph.sum(2))
            #     alphprotein = torch.tanh(alph.sum(1))
            #     alphdrug = alphdrug.unsqueeze(2).expand(-1, -1, 1)
            #     alphprotein = alphprotein.unsqueeze(2).expand(-1, -1, 1)
            #     drug_feature = torch.mul(alphdrug, x.unsqueeze(2)).squeeze(2)
            #     protein_feature = torch.mul(alphprotein, xt.unsqueeze(2)).squeeze(2)
            #     # drug_feature = F.max_pool1d(drug_feature, kernel_size=drug_feature.size(1)).squeeze(2)
            #     # protein_feature = F.max_pool1d(protein_feature, kernel_size=protein_feature.size(1)).squeeze(2)
            #     xc = torch.cat((drug_feature, protein_feature), 1)
            #     idx += 2

            elif name in ['fc1','fc2']:
                w, b = vars[idx], vars[idx + 1]
                xc = F.linear(xc, w, b)
                xc = F.relu(xc)
                xc = self.dropout(xc)
                idx+=2  #15~18
            
            elif name is 'fc3':
                w, b = vars[idx], vars[idx + 1]
                xc = F.linear(xc, w, b)
                #19,20
        return xc       
    
    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    # def message_and_aggregate(self, adj_t: SparseTensor,
    #                           x: OptPairTensor) -> Tensor:
    #     adj_t = adj_t.set_value(None, layout=None)
    #     return matmul(adj_t, x[0], reduce=self.aggr)

    # def named_parameters(self, prefix='', recurse=True):
    #     param_names = ['conv1.bias', 'conv1.lin.weight', 'conv2.bias', 'conv2.lin.weight', 'conv3.bias', 'conv3.lin.weight', 'fc_g1.weight', 'fc_g1.bias', 'fc_g2.weight', 'fc_g2.bias', 'embedding_xt.weight', 'conv_xt_1.weight', 'conv_xt_1.bias', 'fc1_xt.weight', 'fc1_xt.bias', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'out.weight', 'out.bias']
    #     params = dict(super().named_parameters(prefix, recurse))
    #     mapped_params = {name:params[f'vars.{i}'] for i, name in enumerate(param_names)}
    #     return mapped_params.items()