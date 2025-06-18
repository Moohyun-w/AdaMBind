import  torch
import math
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.utils import add_self_loops
from torch_geometric.typing import Adj, OptTensor, PairTensor

from torch import Tensor
from torch_scatter import scatter_add
# from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

config=[
    ('conv1',[78,78]),
    ('conv2', [78*2,78]),
    ('conv3',[78*4,78*2]),
    ('gmp',[True]), 
    ('relu', [True]),
    ('linear1', [1024,78*4]),
    ('linear2', [ 128,1024]),
    ('dropout', [0.]),

    ('embedding', [26,128]),
    ('conv1d',[32,1000,8]),
    ('linear3', [128,32*121,]),

    ('fc1',[ 1024,2*128]),
    ('fc2',[512,1024]),
    ('fc3',[1,512])
]


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    #返回邻接矩阵和D(^0.5) A D(^0.5)

    fill_value = 2. if improved else 1.
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                    device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class GCNNet(MessagePassing):
    def __init__(self):
        super(GCNNet, self).__init__()

        self.vars = nn.ParameterList()
        #GNN层中线性层初始化：weights->kaiming_normal_   ;  bias->zeros
        #普通线性层初始化：weights->kaiming_uniform_   ; bias->uniform_

        for i, (name, param) in enumerate(config):  
            # define a conv2d layer
            if name is 'conv1':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))  #0,1

            elif name is 'conv2':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))  #2,3
            
            elif name is 'conv3':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))  #4,5
            
            elif name is 'linear1':
                # w = nn.Parameter(torch.ones(*param))
                # b = nn.Parameter(torch.zeros(param[0]))
                # torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))
                # fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(w)
                # bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                # torch.nn.init.uniform_(b, -bound, bound)
                # self.vars.append(w)
                # self.vars.append(b)  #6,7
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            
            elif name is 'linear2':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))   #8,9

            elif name is 'embedding':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.normal_(w)
                self.vars.append(w)  #10

            elif name is 'conv1d':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))  #11,12

            elif name is 'linear3':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))  #13,14

            elif name is 'fc1':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))   #17,18
            
            elif name is 'fc2':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))  #19,20

            elif name is 'fc3':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))   #21,22

            elif name is 'dropout':
                self.dropout = nn.Dropout(*param)

            elif name in ['gmp','relu']:
                continue
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, data,vars=None):
        # get graph input
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_weight=None
        c=x.size(0)
        # get protein input
        target = data.target

        if vars is None:
            vars = self.vars

        idx = 0
        for name, param in config:
            #------------------获得药物嵌入-----------------
            if name in ['conv1','conv2','conv3']:
                w, b = vars[idx], vars[idx + 1]
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index,edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]
                x = F.linear(x, w, bias=None)
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
                x=x+b
                x = F.relu(x)
                idx += 2  #0~5

            elif name is 'gmp':
                x = gmp(x, batch)

            elif name is 'relu':
                x=F.relu(x)
            
            elif name in ['linear1','linear2']:
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                x = F.relu(x)
                x = self.dropout(x)
                idx += 2  #6~9
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
            
            elif name is 'linear3':
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
    
    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def named_parameters(self, prefix='', recurse=True):
        param_names = ['conv1.bias', 'conv1.lin.weight', 'conv2.bias', 'conv2.lin.weight', 'conv3.bias', 'conv3.lin.weight', 'fc_g1.weight', 'fc_g1.bias', 'fc_g2.weight', 'fc_g2.bias', 'embedding_xt.weight', 'conv_xt_1.weight', 'conv_xt_1.bias', 'fc1_xt.weight', 'fc1_xt.bias', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'out.weight', 'out.bias']
        params = dict(super().named_parameters(prefix, recurse))
        mapped_params = {name:params[f'vars.{i}'] for i, name in enumerate(param_names)}
        return mapped_params.items()