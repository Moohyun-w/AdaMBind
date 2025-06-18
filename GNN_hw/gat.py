import  torch
from torch import nn
import math
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from torch.nn import functional as F
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import remove_self_loops, add_self_loops,softmax
from torch import Tensor

config=[
    ('conv1',[78,78,10]),
    ('conv2', [128,78*10,1]),
    ('gmp',[True]),
    ('linear1', [128,128]),

    ('embedding', [26,128]),
    ('conv1d',[32,1000,8]),
    ('linear2', [128,32*121,]),

    # ('attention',[128,128]),
    ('fc1',[ 1024,2*128]),
    ('fc2',[256,1024]),
    ('fc3',[1,256]),

    ('relu', [True]),
    ('dropout', [0.])
]


class GATNet(MessagePassing):
    def __init__(self,**kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATNet, self).__init__(node_dim=0, **kwargs)

        self.vars = nn.ParameterList()
        #GNN层中线性层初始化：weights->kaiming_normal_ ;  bias->zeros
        #普通线性层初始化：weights->kaiming_uniform_ ; bias->uniform_

        for i, (name, param) in enumerate(config):  
            # define a conv2d layer
            if name is 'conv1':
                w = nn.Parameter(torch.ones(param[0]*param[2],param[1]))
                att_src=nn.Parameter(torch.ones(1, param[2], param[0]))
                att_dst=nn.Parameter(torch.ones(1, param[2], param[0]))
                torch.nn.init.kaiming_normal_(w)
                torch.nn.init.kaiming_normal_(att_src)
                torch.nn.init.kaiming_normal_(att_dst)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0]*param[2])))  #0,1,2,3
                self.vars.append(att_src)
                self.vars.append(att_dst)
                
            elif name is 'conv2':
                w = nn.Parameter(torch.ones(param[0]*param[2],param[1]))
                att_src=nn.Parameter(torch.ones(1, param[2], param[0]))
                att_dst=nn.Parameter(torch.ones(1, param[2], param[0]))
                torch.nn.init.kaiming_normal_(w)
                torch.nn.init.kaiming_normal_(att_src)
                torch.nn.init.kaiming_normal_(att_dst)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0]*param[2])))  #4,5,6,7
                self.vars.append(att_src)
                self.vars.append(att_dst)
            
            elif name is 'linear1':
                # w = nn.Parameter(torch.ones(*param))
                # b = nn.Parameter(torch.zeros(param[0]))
                # torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))
                # fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(w)
                # bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                # torch.nn.init.uniform_(b, -bound, bound)
                # self.vars.append(w)
                # self.vars.append(b)  #8,9
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0]))) #8,9

            elif name is 'embedding':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.normal_(w)
                self.vars.append(w)  #10

            elif name is 'conv1d':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))  #11,12

            elif name is 'linear2':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))  #13,14
            
            # elif name is 'attention':
            #     w = nn.Parameter(torch.ones(*param))
            #     torch.nn.init.kaiming_normal_(w)
            #     self.vars.append(w)
            #     self.vars.append(nn.Parameter(torch.zeros(param[0])))  #15,16

            elif name is 'fc1':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))  #15,16
            
            elif name is 'fc2':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))  #17,18

            elif name is 'fc3':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))  #19,20

            elif name is 'dropout':
                self.dropout = nn.Dropout(*param)

            elif name in ['gmp','relu']:
                continue      
        self._alpha=None

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
            if name in ['conv1','conv2']:
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

                x = self.propagate(edge_index, x=x, alpha=alpha, size=None)

                alpha = self._alpha
                assert alpha is not None
                self._alpha = None
                
                x = x.view(-1, H * C)
                x = x + b
                x=F.relu(x)
                idx += 4  #0~3,4~7

            elif name is 'gmp':
                x = gmp(x, batch)
            
            elif name in ['linear1']:
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                x = F.relu(x)
                idx += 2  #8,9
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

            #------------预测--------------
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
    
    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # Given egel-level attention coefficients for source and target nodes,
        # we simply need to sum them u0.p to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=0., training=self.training)
        return x_j * alpha.unsqueeze(-1)

    # def named_parameters(self, prefix='', recurse=True):
    #     param_names = ['conv1.bias', 'conv1.lin.weight', 'conv2.bias', 'conv2.lin.weight', 'conv3.bias', 'conv3.lin.weight', 'fc_g1.weight', 'fc_g1.bias', 'fc_g2.weight', 'fc_g2.bias', 'embedding_xt.weight', 'conv_xt_1.weight', 'conv_xt_1.bias', 'fc1_xt.weight', 'fc1_xt.bias', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'out.weight', 'out.bias']
    #     params = dict(super().named_parameters(prefix, recurse))
    #     mapped_params = {name:params[f'vars.{i}'] for i, name in enumerate(param_names)}
    #     return mapped_params.items()
    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)