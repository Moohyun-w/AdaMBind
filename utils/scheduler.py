import torch.nn as nn
from torch.distributions.categorical import Categorical
from collections import OrderedDict
import torch


class Scheduler(nn.Module):
    def __init__(self, N, buffer_size, grad_indexes, use_deepsets=True, simple_loss=False):
        super(Scheduler, self).__init__()
        self.percent_emb = nn.Embedding(100, 5)

        self.grad_lstm = nn.LSTM(N, 10, 1, bidirectional=True)
        self.loss_lstm = nn.LSTM(1, 10, 1, bidirectional=True)
        self.buffer_size = buffer_size
        self.grad_indexes = grad_indexes
        self.use_deepsets = use_deepsets
        self.simple_loss = simple_loss
        self.cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)
        input_dim = 45

        if use_deepsets:
            self.h = nn.Sequential(nn.Linear(input_dim, 20), nn.Tanh(), nn.Linear(20, 10))
            self.fc1 = nn.Linear(input_dim + 10, 20)
        else:
            self.fc1 = nn.Linear(input_dim, 20)
        self.fc2 = nn.Linear(20, 1)
        self.tasks = []

    def forward(self, l, input, pt):
        x_percent = self.percent_emb(pt)  #15*5

        loss_output, (hn, cn) = self.loss_lstm(l.reshape(1, len(l), 1)) #1*15*1 -> 1*15*20
        loss_output = loss_output.sum(0) #15*20
        input = input[0] #15*(21*1)
        grad_output, (hn, cn) = self.grad_lstm(input.reshape(1, len(input), -1))  #15*(21*1) -> 1*15*20
        grad_output = grad_output.sum(0)   #15*20

        x = torch.cat((x_percent, grad_output, loss_output), dim=1)  #15*5,15*20,15*20  #15,45

        if self.use_deepsets:
            x_C = (torch.sum(x, dim=1).unsqueeze(1) - x) / (len(x) - 1)
            x_C_mapping = self.h(x_C)
            x = torch.cat((x, x_C_mapping), dim=1)
            z = torch.tanh(self.fc1(x))
        else:
            z = torch.tanh(self.fc1(x))
        z = self.fc2(z)
        return z

    def add_new_tasks(self, tasks):
        self.tasks.extend(tasks)
        if len(self.tasks) > self.buffer_size:
            self.tasks = self.tasks[-self.buffer_size:]

    def sample_task(self, prob, size, replace=True):
        self.m = Categorical(prob)
        actions = []
        for i in range(size):
            action = self.m.sample()
            if not replace:
                while action in actions:
                    action = self.m.sample()
            actions.append(action)
        actions=[int(it.cpu()) for it in actions]
        return actions

    def compute_loss(self, selected_tasks_idx, maml):
        task_losses = []
        for task_idx in selected_tasks_idx:
            x1, y1, x2, y2 = self.tasks[task_idx]
            x1, y1, x2, y2 = x1.squeeze(0).float().cuda(), y1.squeeze(0).float().cuda(), \
                             x2.squeeze(0).float().cuda(), y2.squeeze(0).float().cuda()
            loss_val = maml(x1, y1, x2, y2)
            task_losses.append(loss_val)
        return torch.stack(task_losses)
    #传入已经训练好的maml，以及外循环pt
    def get_weight(self, task_loss, task_grad_support,task_grad_query, pt,detach=False, task_losses = None, return_grad=False):
        task_acc = []
        task_losses_new = []
        input_embedding = []
        task_grad = [task_grad_query[i] + task_grad_support[i] for i in range(len(task_grad_query))]
        for id in range(len(task_grad)):
            task_g_spt,task_g_qry=task_grad_support[id],task_grad_query[id]
            task_layer_wise_mean_grad = []
            for i in range(len(task_g_qry)):
                if i in self.grad_indexes: 
                    task_layer_wise_mean_grad.append(self.cosine(task_g_spt[i].flatten().unsqueeze(0), task_g_qry[i].flatten().unsqueeze(0)))   #计算spt和qry上模型梯度的余弦相似度，共有6个相似度
            task_layer_wise_mean_grad = torch.stack(task_layer_wise_mean_grad)
            input_embedding.append(task_layer_wise_mean_grad.detach())  
        
        task_loss=[torch.stack(task_loss).cuda()][0]  
        self.task_qry_loss=[d.detach().item() for d in task_loss]
        # if task_loss is None:
        #     task_loss = torch.stack(task_losses_new)
        task_layer_inputs = [torch.stack(input_embedding).cuda()]   
        self.cos_sims=task_layer_inputs[0].squeeze(2).mean(dim=1)
        self.cos_sims=[d.detach().item() for d in self.cos_sims]

        weight = self.forward(task_loss, task_layer_inputs,  
                              torch.tensor([pt]).long().repeat(len(task_loss)).cuda()) 
        if detach: weight = weight.detach()

        if return_grad:
            return task_losses, task_acc, weight, task_layer_inputs
        else:
            return task_losses, task_acc, weight  #weight.size=1*n_buffers