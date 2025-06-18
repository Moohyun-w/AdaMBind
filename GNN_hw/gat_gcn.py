import  torch
from torch import nn
import math
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from torch.nn import functional as F
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import remove_self_loops, add_self_loops,softmax
from torch import Tensor
from torch_scatter import scatter_add
# from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

config=[
    ('conv1',[78,78,10]),
    ('conv2', [780,780]),
    ('pooling',[True]),
    ('linear1', [1500,780*2]),
    ('linear2', [128,1500]),

    ('embedding', [26,128]),
    ('conv1d',[32,1000,8]),
    ('linear3', [128,32*121]),

    # ('attention',[128,128]),
    ('fc1',[1024,256]),
    ('fc2',[512,1024]),
    ('fc3',[1,512]),

    ('relu', [True]),
    ('dropout', [0.])
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

class GAT_GCN(MessagePassing):
    def __init__(self,**kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GAT_GCN, self).__init__(node_dim=0, **kwargs)

        self.vars = nn.ParameterList()
        #GNN层中线性层初始化：weights->kaiming_normal_ ;  bias->zeros
        #普通线性层初始化：weights->kaiming_uniform_ ; bias->uniform_

        for i, (name, param) in enumerate(config):  
            # define a conv2d layer
            if name=='conv1':
                w = nn.Parameter(torch.ones(param[0]*param[2],param[1]))
                att_src=nn.Parameter(torch.ones(1, param[2], param[0]))
                att_dst=nn.Parameter(torch.ones(1, param[2], param[0]))
                torch.nn.init.kaiming_normal_(w)
                torch.nn.init.kaiming_normal_(att_src)
                torch.nn.init.kaiming_normal_(att_dst)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0]*param[2])))  #0,1,2,3
                self.vars.append(att_src)
                self.vars.append(att_dst)  #0~3
                
            elif name=='conv2':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0]))) #4,5
            
            elif name == 'linear1':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0]))) #6,7

            elif name == 'linear2':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0]))) #8,9

            elif name == 'embedding':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.normal_(w)
                self.vars.append(w)  #10

            elif name == 'conv1d':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))  #11,12

            elif name == 'linear3':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0]))) #13,14

            # elif name is 'attention':
            #     w = nn.Parameter(torch.ones(*param))
            #     torch.nn.init.kaiming_normal_(w)
            #     self.vars.append(w)
            #     self.vars.append(nn.Parameter(torch.zeros(param[0])))  #15,16

            elif name == 'fc1':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0]))) #15,16
            
            elif name == 'fc2':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0]))) #17,18

            elif name == 'fc3':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))  #19,20

            elif name == 'dropout':
                self.dropout = nn.Dropout(*param)

            elif name in ['pooling','relu']:
                continue      
        self._alpha=None
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, data,vars=None):
        # get graph input
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_weight=None
        # get protein input
        target = data.target

        if vars is None:
            vars = self.vars

        idx = 0
        for name, param in config:
            #------------------获得药物嵌入-----------------
            if name == 'conv1':
                x = F.dropout(x, p=0.2, training=self.training)

                w, b = vars[idx], vars[idx + 1]
                att_src,att_dst=vars[idx+2],vars[idx+3]
                H, C = param[-1], param[0]

                x_src = x_dst=F.linear(x, w).view(-1,H,C)
                x = (x_src, x_dst)

                alpha_src = (x_src * att_src).sum(dim=-1)
                alpha_dst = None if x_dst is None else (x_dst * att_dst).sum(-1)
                alpha = (alpha_src, alpha_dst)

                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

                x = self.propagate(edge_index, x=x, alpha=alpha, size=None,edge_weight=torch.ones(edge_index.shape[1], device=edge_index.device))
        
                alpha = self._alpha
                assert alpha is not None
                self._alpha = None
                
                x = x.view(-1, H * C)
                x = x + b
                x=F.relu(x)
                idx += 4  #0~3

            elif name == 'conv2':
                w, b = vars[idx], vars[idx + 1]
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index,edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]
                x = F.linear(x, w, bias=None)
                x = self.propagate(edge_index, x=x, alpha=None, size=None,edge_weight=None)
                x=x+b
                x = F.relu(x)
                idx += 2 #4,5
            
            elif name == 'pooling':
                x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
            
            elif name == 'linear1':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                x = F.relu(x)
                idx += 2  #6,7
            
            elif name == 'linear2':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2  #8,9
            #---------------------------------------

            #-------------获得靶蛋白嵌入-------------
            elif name == 'embedding':
                w= vars[idx]
                embedded_xt = F.embedding(target,w)
                idx+=1  #10
            
            elif name == 'conv1d':
                w, b = vars[idx], vars[idx + 1] 
                xt=F.conv1d(embedded_xt,w,b).view(-1, 32 * 121)
                idx+=2  #11,12
            
            elif name == 'linear3':
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
                
            #------------预测--------------
            elif name in ['fc1','fc2']:
                w, b = vars[idx], vars[idx + 1]
                xc = F.linear(xc, w, b)
                xc = F.relu(xc)
                xc = self.dropout(xc)
                idx+=2  #15~18
            
            elif name == 'fc3':
                w, b = vars[idx], vars[idx + 1]
                xc = F.linear(xc, w, b)
                #19,20
        return xc       
    
    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int],edge_weight=OptTensor) -> Tensor:
        if alpha_j is None:
            return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
        else:
            # Given egel-level attention coefficients for source and target nodes,
            # we simply need to sum them u0.p to "emulate" concatenation:
            alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

            alpha = F.leaky_relu(alpha, 0.2)
            alpha = softmax(alpha, index, ptr, size_i)
            self._alpha = alpha  # Save for later use.
            alpha = F.dropout(alpha, p=0., training=self.training)
            return x_j * alpha.unsqueeze(-1)
        
    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)