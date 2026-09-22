"""From-scratch joint prefix reference for the shared ResNet architecture."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from cl_base import ContinualMethod


class GlobalLabels(Dataset):
    def __init__(self,dataset,offset=0): self.dataset,self.offset=dataset,offset
    def __len__(self): return len(self.dataset)
    def __getitem__(self,index):
        x,y=self.dataset[index][:2];return x,y+self.offset


class JointLoss(nn.Module):
    def __init__(self,scenario,width=5):super().__init__();self.scenario,self.width=scenario,width
    def forward(self,logits,y):
        if self.scenario=='class_il':return F.cross_entropy(logits,y)
        return sum(F.cross_entropy(logits[y//self.width==t,t*self.width:(t+1)*self.width],
            y[y//self.width==t]%self.width,reduction='sum') for t in (y//self.width).unique().tolist())/len(y)


class JointPrefix(ContinualMethod):
    name='joint'
    def __init__(self,*args,**kw):
        super().__init__(*args,**kw)
        with self.model.full_output_space(10): pass
        self.initial=copy.deepcopy(self.model).cpu()
        self.criterion=JointLoss(self.scenario)
        self.train_sets=[];self.val_sets=[]
    def logits(self,x,task_id,for_training=True):return self.model(x)
    def train_task(self,task_id,train_loader,val_loader,num_epochs=200,patience=20,verbose=True):
        self.model=copy.deepcopy(self.initial).to(self.device)
        self.model.ensure_head(task_id)
        offset=task_id*5 if self.scenario=='task_il' else 0
        self.train_sets.append(GlobalLabels(train_loader.dataset,offset))
        self.val_sets.append(GlobalLabels(val_loader.dataset,offset))
        train=DataLoader(ConcatDataset(self.train_sets),batch_size=train_loader.batch_size,shuffle=True,num_workers=2)
        val=DataLoader(ConcatDataset(self.val_sets),batch_size=val_loader.batch_size,num_workers=2)
        result=super().train_task(task_id,train,val,num_epochs,patience,verbose)
        result.update(joint_prefix_tasks=task_id+1,training_examples=len(train.dataset),
                      variant='from-scratch joint prefix, shared ResNet18')
        return result
