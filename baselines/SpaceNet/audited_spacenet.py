"""SpaceNet MLP adapted from the authors' MNIST code to the GTEP halves.

Preserves selected-node pools, neuron reservation, weight importance, and
drop/grow. This is explicitly a CIFAR architecture adaptation, not a published
CIFAR result. Reference: third_party/spacenet/{models.py,CL.py}.
"""
import copy
import torch
from torch import nn
from method_loader import load as _load
SpaceNet = _load('baselines/SpaceNet', 'spacenet').SpaceNet
from wsn_helpers import _maskable_modules


class SpaceNetMLP(nn.Module):
    # SpaceNet is published as an MLP.  The defaults are CIFAR-100's GTEP half
    # (3x32x32 in, 50 classes per half, 5 per task); the other datasets pass
    # their own geometry.
    def __init__(self,in_dim=3072,num_classes=50,classes_per_task=5):
        super().__init__()
        self.layers=nn.ModuleList([nn.Linear(in_dim,400),nn.Linear(400,400),
                                   nn.Linear(400,num_classes)])
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight);nn.init.zeros_(layer.bias)
        self.seen_upto=-1;self.classes_per_task=classes_per_task;self.feature_dim=400

    def ensure_head(self, task): self.seen_upto=max(self.seen_upto,task)
    def active_tasks(self): return list(range(self.seen_upto+1))
    def get_features(self,x):
        return self.layers[1](self.layers[0](x.flatten(1)).relu()).relu()
    def forward(self,x,task_id=None):
        out=self.layers[2](self.get_features(x));c=self.classes_per_task
        return out[:,:c*(self.seen_upto+1)] if task_id is None else out[:,task_id*c:(task_id+1)*c]


class AuditedSpaceNet(SpaceNet):
    def __init__(self,*args,density_factor=1.,**kw):
        super().__init__(*args,**kw)
        self.density_factor=density_factor
        # Layer widths come from the model, not from CIFAR-100's geometry, so the
        # same code runs the other benchmarks' input sizes and half widths.
        layers=self.model.layers
        self.widths=(layers[0].in_features,layers[0].out_features,
                     layers[1].out_features,layers[2].out_features)
        self.classes_per_task=getattr(self.model,'classes_per_task',5)
        self.free_nodes=[torch.ones(n,device=self.device,dtype=torch.bool) for n in self.widths]
        self.node_importance=[torch.zeros(n,device=self.device) for n in self.widths]
        self.initial_weights={n:m.parametrizations.weight.original.detach().clone()
                              for n,m in _maskable_modules(self.model)}
        self.allowed={};self.selected=[];self.bias_before={}

    @torch.no_grad()
    def before_task(self,task_id,train_loader,val_loader):
        selected=[self.free_nodes[0].clone()]
        for free in self.free_nodes[1:3]:
            indices=free.nonzero().flatten()
            chosen=indices[torch.randperm(len(indices),device=self.device)[:80]]
            mask=torch.zeros_like(free);mask[chosen]=True;selected.append(mask)
        c=self.classes_per_task
        output=torch.zeros_like(self.free_nodes[3]);output[task_id*c:(task_id+1)*c]=True
        selected.append(output);self.selected=selected
        for v in self.node_importance:v.zero_()
        # The authors' MNIST budgets, the first scaled by the input size.
        budgets=(round(10000*self.widths[0]/784),1640,200)
        for i,(name,module) in enumerate(_maskable_modules(self.model)):
            allowed=selected[i+1][:,None]&selected[i][None,:]&~self.reserved[name]
            indices=allowed.flatten().nonzero().flatten()
            k=min(len(indices),round(budgets[i]*self.density_factor))
            if k==0:raise RuntimeError('SpaceNet exhausted its node/connection pool')
            chosen=indices[torch.randperm(len(indices),device=self.device)[:k]]
            mask=torch.zeros_like(allowed).flatten();mask[chosen]=True
            self.current[name]=mask.reshape_as(allowed);self.allowed[name]=allowed
            module.parametrizations.weight.original[self.current[name]]=self.initial_weights[name][self.current[name]]
            self.parametrisations[name].mask.copy_((self.current[name]|self.reserved[name]).float())
            self.weight_importance[name].zero_()

    def after_backward(self,task_id):
        super().after_backward(task_id)
        for i,(name,module) in enumerate(_maskable_modules(self.model)):
            if module.bias is not None:
                self.bias_before[name]=module.bias.detach().clone()
                free=self.free_nodes[i+1] if i<2 else self.selected[-1]
                if module.bias.grad is not None:module.bias.grad.mul_(free)

    @torch.no_grad()
    def after_step(self,task_id):
        super().after_step(task_id)
        for i,(name,module) in enumerate(_maskable_modules(self.model)):
            importance=self.weight_importance[name]
            self.node_importance[i].add_(importance.sum(0))
            if i==2:self.node_importance[i+1].add_(importance.sum(1))
            if module.bias is not None:
                free=self.free_nodes[i+1] if i<2 else self.selected[-1]
                module.bias[~free]=self.bias_before[name][~free]
        self.bias_before.clear()

    @torch.no_grad()
    def _rewire(self):
        for i,(name,module) in enumerate(_maskable_modules(self.model)):
            if i==2:continue # reference retains the output connections
            active=self.current[name].flatten();importance=self.weight_importance[name].flatten()
            n=min(int(active.sum()*self.rewire_fraction),int(active.sum())-1)
            candidates=(self.allowed[name].flatten()&~active).nonzero().flatten()
            n=min(n,len(candidates))
            if n<=0:continue
            drop=importance.masked_fill(~active,float('inf')).topk(n,largest=False).indices
            endpoint=(self.node_importance[i+1][:,None]*self.node_importance[i][None,:]).flatten()
            grow=candidates[endpoint[candidates].topk(n).indices]
            active[drop]=False;active[grow]=True
            w=module.parametrizations.weight.original
            w.flatten()[drop]=0;w.flatten()[grow]=0
            self.parametrisations[name].mask.copy_((self.current[name]|self.reserved[name]).float())

    def training_state_dict(self):
        return copy.deepcopy(dict(node_importance=self.node_importance,weight_importance=self.weight_importance))
    def load_training_state_dict(self,state):
        if state is not None:
            self.node_importance=state['node_importance'];self.weight_importance=state['weight_importance']

    def after_task(self,task_id,train_loader,val_loader):
        for i in (1,2):
            candidates=self.selected[i].nonzero().flatten()
            chosen=candidates[self.node_importance[i][candidates].topk(min(40,len(candidates))).indices]
            self.free_nodes[i][chosen]=False
        result=super().after_task(task_id,train_loader,val_loader)
        result.update(variant='SpaceNet native MLP adapted to RGB CIFAR halves',
                      free_hidden_neurons=[int(v.sum()) for v in self.free_nodes[1:3]])
        return result
