"""Recorded optimizer/schedule policy; defaults leave legacy runs unchanged."""
import math
import torch


def optimizer_for(owner, parameters):
    policy = getattr(owner, 'training_policy', {})
    kind = policy.get('optimizer', 'adam').lower()
    kw = dict(lr=owner.lr, weight_decay=policy.get('weight_decay', getattr(owner, 'weight_decay', 0.)))
    if kind == 'sgd':
        return torch.optim.SGD(parameters, momentum=policy.get('momentum', .9), **kw)
    if kind == 'adam':
        return torch.optim.Adam(parameters, **kw)
    raise ValueError(f'Unsupported optimizer {kind}')


class EpochPolicy:
    def __init__(self, owner, optimizer, epochs, patience):
        self.policy = getattr(owner, 'training_policy', {})
        self.epochs, self.patience = epochs, patience
        self.minimum = min(epochs, self.policy.get('min_epochs', 0))
        self.scheduler = None
        kind = self.policy.get('scheduler', 'none')
        if kind == 'steplr':
            n = self.policy.get('milestone_count', 2)
            milestones = [int(epochs*2*i/(2*n+1)) for i in range(1,n+1)]
            self.minimum = max(self.minimum, min(epochs, max(milestones)+patience))
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones,
                gamma=self.policy.get('lr_decay', .1))
        elif kind == 'cosine':
            self.minimum = max(self.minimum, math.ceil(.8*epochs))
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        elif kind == 'linear':
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: max(0.,1-e/epochs))
        elif kind != 'none':
            raise ValueError(f'Unsupported schedule {kind}')

    def step(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def should_stop(self, completed, waited):
        return completed >= self.minimum and waited >= self.patience
