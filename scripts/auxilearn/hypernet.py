from abc import abstractmethod

from torch import nn
from torch.nn.utils import weight_norm
import torch
import torch.nn.functional as F


class Naive_hyper(nn.Module):
    def __init__(self, data_num, task_num):
        super(Naive_hyper, self).__init__()
        self.weights = nn.Embedding(data_num, task_num)
        self.nolinear = nn.Softplus()
    
    def forward(self, losses, sample_id):
        current_weight = self.nolinear(self.weights(sample_id))
        final_loss = ( (current_weight*losses).mean(0) ).sum()
        return final_loss

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

class HyperNet(nn.Module):
    def __init__(self, main_task, input_dim):
        super().__init__()
        self.main_task = main_task
        self.input_dim = input_dim

    def forward(self, losses, outputs=None, labels=None, data=None):
        pass

    def _init_weights(self):
        pass

    def get_weights(self):
        return list(self.parameters())


class MonoHyperNet(HyperNet):
    """Monotonic Hypernets

    """
    def __init__(self, main_task, input_dim, clamp_bias=False):
        super().__init__(main_task=main_task, input_dim=input_dim)
        self.clamp_bias = clamp_bias

    def get_weights(self):
        return list(self.parameters())

    @abstractmethod
    def clamp(self):
        pass


class MonoJoint(MonoHyperNet):

    def __init__(self, main_task, input_dim, device, nonlinearity=None, bias=True, dropout_rate=0., weight_normalization=True, K = 2, init_lower= 0.0, init_upper=1.0):
        super().__init__(main_task = main_task, input_dim = input_dim) # 

        self.device = device
        self.nonlinearity = nonlinearity if nonlinearity is not None else nn.Softplus()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.weight_normalization = weight_normalization
        self.direction_a = nn.Parameter(torch.torch.rand(1,input_dim).to(torch.device(device=self.device)), requires_grad=True)#torch.rand
        
        self.causal_effect = nn.Parameter(torch.torch.rand(1,input_dim).to(torch.device(device=self.device)), requires_grad=True)#torch.ones
        
        self.layers = []
        self.net = nn.Sequential(*self.layers)
        self.input_dim = input_dim
    
    def norm_loss1(self,losses):
        ###### this performs bad
        m = losses.mean(dim=0,keepdim = True)
        std = losses.std(0, keepdim=True)
        return (losses - m) / (std + 1e-6)

    @staticmethod
    def _init_layer(layer, init_lower, init_upper):
        b = init_upper if init_upper is not None else 1.
        a = init_lower if init_lower is not None else 0.
        if isinstance(layer, nn.Linear):
            layer.weight = nn.init.uniform_(layer.weight, b=b, a=a)
            if layer.bias is not None:
                layer.bias = nn.init.constant_(layer.bias, 0.)
    
    def get_loss_weight(self, dir_matrix, mag_matrix):
        
        dir_matrix = torch.stack(dir_matrix)
        
        dir = self.nonlinearity(dir_matrix * self.direction_a)
        
        return dir 

    def recorder(self,output_content):
        file_path = "./thin_scheduling_10_log.txt"
        with open(file_path, 'a') as f:
            f.write(output_content + '\n')


    def forward(self, causal_effect, dir_matrix, mag_matrix, losses, to_train = True):

        inter_task_sched = self.get_loss_weight(dir_matrix,mag_matrix)

        #causal_effect = torch.stack(causal_effect)
        #prompt_sched = self.nonlinearity(causal_effect * self.causal_effect)

        losses = torch.stack(losses)
        sched = inter_task_sched #* prompt_sched

        if not to_train:
            total_sched = sched

            return ((total_sched * losses).sum())
        else:

            #dir_matrix=torch.unsqueeze(torch.stack(dir_matrix), dim=0)
            '''
            self.recorder('dir_matrix: ' + str(dir_matrix))
            self.recorder('task affinity scheduing weight: ' + str(self.direction_a))
            self.recorder('task scheduling: ' + str(inter_task_sched.detach()))

            self.recorder('causal effect: ' + str(causal_effect))
            self.recorder('causal effect scheduing weight: ' + str(self.causal_effect))
            self.recorder('prompt scheduling: ' + str(prompt_sched.detach()))
            '''
            total_sched = sched.detach()
            sum_total_sched = torch.sum(total_sched)
            total_sched = (self.input_dim * total_sched) / sum_total_sched
            #self.recorder('total_sched: ' + str(total_sched))
            #print("total_sched", total_sched)

            return ((total_sched * losses).sum())


    def clamp(self):
        for l in self.net:
            if isinstance(l, nn.Linear):
                if self.weight_normalization:
                    l.weight_v.data.clamp_(0)
                    l.weight_g.data.clamp_(0)
                else:
                    l.weight.data.clamp_(0)

                if l.bias is not None and self.clamp_bias:
                    l.bias.data.clamp_(0)

