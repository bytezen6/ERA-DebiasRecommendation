import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.autograd import Variable
from typing import * 


def to_var(x:Tensor, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class GradReverseLayer(torch.autograd.Function):
    """
    Rewrite custom gradient calculation method
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
       for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p
    
    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self,curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)
            
    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())   
                
    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, in_features, out_features, bias):
        super().__init__()
        ignore = nn.Linear(in_features, out_features, bias)
       
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
    
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaEmbed(MetaModule): 
    def __init__(self, num, dim):
        super().__init__()
        ignore = nn.Embedding(num, dim)
        
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', None)
        
    def forward(self, x):
        return self.weight[x]
    
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]
    

class dc2(nn.Module):
    def __init__(self, input_dim, mid_dim):
        super(dc2, self).__init__()
        self.emb_dim = mid_dim
        self.input_dim = input_dim

        self.bottle_trans = nn.Parameter(torch.zeros(self.input_dim, self.emb_dim))
        self.bottle_bias = nn.Parameter(torch.zeros(1, self.emb_dim))

        self.classifier_trans1 = nn.Parameter(torch.zeros(self.emb_dim, self.emb_dim))
        self.classifier_bias1 = nn.Parameter(torch.zeros(1, self.emb_dim))
        self.classifier_trans2 = nn.Parameter(torch.zeros(self.emb_dim, 1))
        self.classifier_bias2 = nn.Parameter(torch.zeros(1))

        self.non_linear = nn.Softplus()
        nn.init.xavier_normal_(self.bottle_trans)
        nn.init.xavier_normal_(self.classifier_trans1)
        nn.init.xavier_normal_(self.classifier_trans2)

        return 
    
    def forward(self, event_emb, dt):
        prob = self.mlp(event_emb, dt).squeeze()

        return prob

    def mlp(self, event_emb, dt):
        if dt:
            bt_emb = torch.matmul(event_emb, self.bottle_trans.detach()) + self.bottle_bias.detach()
            bt_emb = self.non_linear(bt_emb)

            ct_emb1 = torch.matmul(bt_emb, self.classifier_trans1.detach()) + self.classifier_bias1.detach()
            ct_emb1 = self.non_linear(ct_emb1)
            rating = torch.matmul(ct_emb1, self.classifier_trans2.detach()) + self.classifier_bias2.detach()
        else:
            bt_emb = torch.matmul(event_emb, self.bottle_trans) + self.bottle_bias
            bt_emb = self.non_linear(bt_emb)

            ct_emb1 = torch.matmul(bt_emb, self.classifier_trans1) + self.classifier_bias1
            ct_emb1 = self.non_linear(ct_emb1)
            rating = torch.matmul(ct_emb1, self.classifier_trans2) + self.classifier_bias2

        prob = torch.sigmoid(rating)

        return prob



class DisBC(MetaModule):
    def __init__(self, user_num, item_num, dim):
        super().__init__()
        self.user_embedding = MetaEmbed(user_num, dim)
        self.item_embedding = MetaEmbed(item_num, dim)

        self.fn = nn.Dropout(0.01)

        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, )
                nn.init.zeros_(m.bias) 


    def get_z(self, user, item, if_drop=True)->Tensor:
        z:Tensor = self.user_embedding(user) * self.item_embedding(item)
        # if self.training:
        #     z += torch.randn_like(z).cuda() * 1e-3

        return z
    
    def forward(self, user, item)->Tensor:
        z = self.get_z(user, item)
        # z = self.fn(z)
        return self.from_z_to_predict(z)

    def from_z_to_predict(self, z:Tensor)->Tensor:
        return z.sum(dim=1).view(-1).sigmoid()

    predict = forward 




class zw(nn.Module):
    def __init__(self, dim):
        super().__init__()
        temp = torch.tensor([1.0, 0.1, 0.1]).cuda()
        self.weight = nn.Parameter(temp, requires_grad=True)
        # self.register_parameter('weight', to_var(temp, requires_grad=True))
        

    def forward(self,x):
        # print(self.weight.data)
        return self.weight[x].exp().view(-1)

    

class zw2(nn.Module):
    def __init__(self, n_user, n_item, n_dim):
        super().__init__()
        self.u = nn.Parameter(torch.zeros((n_user,) ), requires_grad=True).cuda()
        self.i = nn.Parameter(torch.zeros((n_item,) ), requires_grad=True).cuda()
        self.z = nn.Parameter(torch.tensor([1.0, 0.1, 0.1]).cuda(), requires_grad=True).cuda()


    def forward(self, uid, iid, r):
        # print(self.z.data)
        z =  self.u[uid].view(-1) + self.i[iid].view(-1) + self.z[r].view(-1)
        return z.exp().view(-1)




class DisBC_ARGS(MetaModule):
    def __init__(self, user_num, item_num, dim, args):
        super().__init__()
        self.user_embedding = MetaEmbed(user_num, dim)
        self.item_embedding = MetaEmbed(item_num, dim)
        self.args = args
        self.fn = nn.Dropout(0.01)

        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, )
                nn.init.zeros_(m.bias) 


    def get_z(self, user, item, if_drop=True)->Tensor:
        z:Tensor = self.user_embedding(user) * self.item_embedding(item)
        if self.training:
            z += torch.randn_like(z).cuda() * self.args["noise"]

        return z
    
    def forward(self, user, item)->Tensor:
        z = self.get_z(user, item)
        # z = self.fn(z)
        return self.from_z_to_predict(z)

    def from_z_to_predict(self, z:Tensor)->Tensor:
        return z.sum(dim=1).view(-1).sigmoid()

    predict = forward 