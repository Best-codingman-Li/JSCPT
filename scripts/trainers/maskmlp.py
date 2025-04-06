import torch
import torch.nn as nn
import torch.optim as optim


class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.mask = nn.Parameter(torch.ones(out_features, in_features))  # 初始化为全1的mask

    def forward(self, x):
        masked_weight = self.weight * self.mask
        return F.linear(x, masked_weight, self.bias)


class MaskMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super(MaskMLP, self).__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.mask = nn.Parameter(torch.zeros(in_dim, requires_grad=True))
        self.maskfc = nn.Linear(in_dim, in_dim)
        self.use_residual = use_residual
        # self.act_fn = nn.GELU()
        self.act_fn = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mask = self.sigmoid(self.mask).unsqueeze(0)
        cf_x = torch.mul(x, 1 - mask)
        cf_residual = cf_x

        x = torch.mul(x, mask)
        residual = x

        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        #x = self.act_fn(x)

        cf_x = self.fc1(cf_x)
        cf_x = self.act_fn(cf_x)
        cf_x = self.fc2(cf_x)
        #cf_x = self.act_fn(cf_x)

        if self.use_residual:
            x = x + residual
            cf_x = cf_x + cf_residual
        return x, cf_x


class MergeMaskMLP(nn.Module):
    def __init__(self, subject_num, in_dim, out_dim, hidden_dim, use_residual=True):
        super(MergeMaskMLP, self).__init__()
        if use_residual:
            assert in_dim == out_dim
        self.subject_num = subject_num
        self.layernorm = nn.LayerNorm(in_dim)

        self.singleResmlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))
        self.mergeResmlp = nn.Sequential(nn.Linear(self.subject_num * in_dim, self.subject_num * hidden_dim * 2), nn.ReLU(), nn.Linear(self.subject_num * hidden_dim * 2, self.subject_num * hidden_dim), nn.ReLU(), nn.Linear(self.subject_num * hidden_dim, out_dim))
        self.mask = [nn.Parameter(torch.ones(in_dim, requires_grad=True)) for _ in range(self.subject_num)]
        self.maskfc = [nn.Linear(in_dim, in_dim) for _ in range(self.subject_num)]
        self.use_residual = use_residual
        # self.act_fn = nn.GELU()
        self.act_fn = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        masks = [self.sigmoid(self.mask[i]).unsqueeze(0).cuda() for i in range(self.subject_num)]

        x_list = [torch.mul(x, masks[i]) for i in range(self.subject_num)]
        residual_list = x_list

        cf_x_list = [torch.mul(x, 1 - masks[i]) for i in range(self.subject_num)]
        cf_x_list = cf_x_list

        x_list = [self.singleResmlp(x_list[i]) for i in range(self.subject_num)]
        cf_x_list = [self.singleResmlp(cf_x_list[i]) for i in range(self.subject_num)]
        if self.use_residual:
            out_x_list = [(x_list[i] + residual_list[i]) for i in range(self.subject_num)]
            out_cf_x_list = [(cf_x_list[i] + cf_x_list[i]) for i in range(self.subject_num)]

        merge_x = torch.cat(x_list, dim=2)
        #print("merge_x.size()", merge_x.size())
        merge_cf = torch.cat(cf_x_list, dim=2)

        out_merge_x = self.mergeResmlp(merge_x) 
        out_merge_cf = self.mergeResmlp(merge_cf)

        return out_x_list, out_cf_x_list, out_merge_x, out_merge_cf


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x
