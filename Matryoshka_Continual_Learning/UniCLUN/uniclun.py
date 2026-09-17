"""UniCLUN paper-based CL adapter, arXiv:2408.11374 equations 2--8.

Official CLMUL lacks its imported ocdnet implementation. Explicit choices here:
negative contrastive log-likelihood, mean over positive pairs, a two-layer
128-D projector, and copied student BN buffers on stochastic teacher updates.
This is a labeled reimplementation, not verified official-code parity.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from cl_base import ContinualMethod, Reservoir


def contrastive(a, b, labels, temperature, exclude_self=False):
    a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
    scores = a @ b.T / temperature
    positive = labels[:, None].eq(labels[None, :])
    if exclude_self:
        eye = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
        positive = positive & ~eye
        scores = scores.masked_fill(eye, -1e9)
    valid = positive.any(1)
    if not valid.any():
        return a.sum() * 0
    return -(F.log_softmax(scores, dim=1) * positive).sum(1).div(
        positive.sum(1).clamp_min(1))[valid].mean()


class UniCLUN(ContinualMethod):
    name = 'uniclun'

    def __init__(self, model, device, *args, buffer_size=1000, rho=2.,
                 contrastive_temperature=.1, alpha1=1., alpha2=1., alpha3=1.,
                 momentum=.99, bernoulli_p=.6, projector_dim=128, **kw):
        super().__init__(model, device, *args, **kw)
        width = model.feature_dim
        self.model.uniclun_projector = nn.Sequential(nn.Linear(width, width), nn.ReLU(),
                                                     nn.Linear(width, projector_dim)).to(device)
        self.student = None
        self.buffer = Reservoir(buffer_size, device)
        self.rho, self.tau = rho, contrastive_temperature
        self.alpha1, self.alpha2, self.alpha3 = alpha1, alpha2, alpha3
        self.momentum, self.bernoulli_p = momentum, bernoulli_p

    @torch.no_grad()
    def update_teacher(self):
        if torch.rand((), device=self.device) >= self.bernoulli_p:
            return False
        for target, source in zip(self.model.parameters(), self.student.parameters()):
            target.lerp_(source, 1 - self.momentum)
        for target, source in zip(self.model.buffers(), self.student.buffers()):
            target.copy_(source)
        return True

    def _outputs(self, model, x):
        features = model.get_features(x)
        logits = torch.cat([model.heads[str(t)](features) for t in model.active_tasks()], 1)
        return logits, model.uniclun_projector(features)

    def train_task(self, task_id, train_loader, val_loader, num_epochs=50,
                   patience=10, verbose=True):
        from tqdm import tqdm
        self.model.ensure_head(task_id)
        self.model.to(self.device)
        if self.student is None:
            self.student = copy.deepcopy(self.model)
        else:
            self.student.ensure_head(task_id)
            self.student.to(self.device)
            self.student.heads[str(task_id)].load_state_dict(self.model.heads[str(task_id)].state_dict())
        self.student.requires_grad_(True)
        self.model.requires_grad_(False)
        self.model.eval()
        from training_policy import optimizer_for, EpochPolicy
        opt = optimizer_for(self, self.student.parameters())
        schedule = EpochPolicy(self, opt, num_epochs, patience)
        epoch_log = []
        best, best_state, waited, updates, epochs_run = float('inf'), None, 0, 0, 0
        for _ in tqdm(range(num_epochs), desc=f'UniCLUN task {task_id}', disable=not verbose):
            epochs_run += 1
            self.student.train()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                if self.scenario == 'task_il':
                    y = y + task_id * self.model.classes_per_task
                n_current = len(y)
                replay = self.buffer.sample(n_current)
                if replay is not None:
                    x, y = torch.cat((x, replay[0])), torch.cat((y, replay[1]))
                opt.zero_grad(set_to_none=True)
                logits, projection = self._outputs(self.student, x)
                with torch.no_grad():
                    target, target_projection = self._outputs(self.model, x)
                if self.scenario == 'class_il':
                    ce = F.cross_entropy(logits, y)
                else:
                    width = self.model.classes_per_task
                    ce = sum(F.cross_entropy(logits[y // width == int(t), int(t)*width:(int(t)+1)*width],
                              y[y // width == int(t)] % width, reduction='sum')
                             for t in (y // width).unique()) / len(y)
                od = logits.sum() * 0
                if replay is not None:
                    confidence = F.softmax(target[n_current:] / self.rho, 1).gather(
                        1, y[n_current:, None]).squeeze(1)
                    od = (confidence * (logits[n_current:] - target[n_current:]).square().sum(1)).mean()
                loss = (ce + self.alpha1 * od + self.alpha2 * contrastive(
                    projection, target_projection, y, self.tau) + self.alpha3 * contrastive(
                    projection, projection, y, self.tau, exclude_self=True))
                loss.backward()
                opt.step()
                updates += int(self.update_teacher())
            val, _ = self.validate(val_loader, task_id)
            epoch_log.append(dict(epoch=epochs_run, validation_loss=val))
            if val < best:
                best, waited = val, 0
                best_state = (copy.deepcopy(self.model.state_dict()), copy.deepcopy(self.student.state_dict()))
            else:
                waited += 1
                if schedule.should_stop(epochs_run, waited):
                    break
            schedule.step()
        if best_state is not None:
            self.model.load_state_dict(best_state[0])
            self.student.load_state_dict(best_state[1])
        for x, y in train_loader:
            if self.scenario == 'task_il':
                y = y + task_id * self.model.classes_per_task
            self.buffer.add(x, y, task_id=task_id)
        result = {'task_id': task_id, 'epochs_run': epochs_run, 'teacher_updates': updates,
                  'epoch_log': epoch_log, 'epoch_cap': num_epochs,
                  'cap_reached_while_improving': epochs_run==num_epochs and waited<patience,
                  'buffer_examples': len(self.buffer), 'variant': 'UniCLUN paper-based CL adapter'}
        self.history.append(result)
        return result
