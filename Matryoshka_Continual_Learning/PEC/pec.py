"""PEC adapted to the GTEP protocol from the official implementation.

Mechanism ported verbatim from github.com/michalzajac-ml/pec @ 3c15633
(models/pec.py, backbone/pec_modules.py):

  * one frozen teacher shared by every class (upstream passes a single
    ``teacher`` instance into every PecStudentTeacherPair),
  * per-class students that map images directly to scores -- no shared
    backbone, no shared features (upstream asserts ``backbone is None``),
  * every student initialised to student[0]'s weights,
  * score for class c is the negative student/teacher squared error, so
    argmax over classes is the prediction,
  * one optimizer per student; a class is only ever touched while its own
    task is being trained, so old students are bit-exact afterwards.

GTEP deviations (deliberate, so PEC is comparable to the other baselines):
five-class tasks instead of one-class, multi-epoch training with validation
early stopping instead of the published single online pass, and the campaign's
tuned learning rate instead of the fixed 0.001.
"""
import copy
import torch
from torch import nn
from cl_base import ContinualMethod

# Official CIFAR-100 settings: --pec_conv_layers="(40, 3, 1)" --pec_output_dim=172
# --pec_conv_reduce_spatial_to=4 --pec_teacher_width_multiplier=100
CONV_LAYERS = ((40, 3, 1),)
OUTPUT_DIM = 172
REDUCE_SPATIAL_TO = 4
TEACHER_MULTIPLIER = 100


def pec_network(conv_layers=CONV_LAYERS, output_dim=OUTPUT_DIM,
                reduce_spatial_to=REDUCE_SPATIAL_TO, is_teacher=False,
                teacher_multiplier=TEACHER_MULTIPLIER, in_channels=3):
    """PecCNN from backbone/pec_modules.py."""
    layers, cur = [], in_channels
    for channels, kernel, stride in conv_layers:
        if is_teacher:
            channels = int(teacher_multiplier * channels)
        layers += [nn.Conv2d(cur, channels, kernel, stride=stride, padding=kernel // 2)]
        cur = channels
        layers += [nn.InstanceNorm2d(cur, affine=True), nn.ReLU()]
    layers += [nn.AdaptiveAvgPool2d(reduce_spatial_to), nn.Flatten(1),
               nn.Linear(cur * reduce_spatial_to ** 2, output_dim)]
    return nn.Sequential(*layers)


class PECNet(nn.Module):
    def __init__(self, num_classes=50, classes_per_task=5):
        super().__init__()
        self.classes_per_task = classes_per_task
        self.num_classes = num_classes
        self.feature_dim = OUTPUT_DIM
        # A single frozen teacher shared by every class, as upstream does.
        self.teacher = pec_network(is_teacher=True)
        self.teacher.requires_grad_(False)
        initial = pec_network()
        self.students = nn.ModuleList([copy.deepcopy(initial) for _ in range(num_classes)])
        for student in self.students:  # identical init, as upstream enforces
            student.load_state_dict(self.students[0].state_dict())
        self.seen_upto = -1

    def ensure_head(self, task_id):
        self.seen_upto = max(self.seen_upto, task_id)

    def active_tasks(self):
        return list(range(self.seen_upto + 1))

    def pec_error(self, x, classes):
        """Mean squared student/teacher error per class (upstream forward)."""
        with torch.no_grad():
            target = self.teacher(x)
        return torch.stack([(self.students[c](x) - target).square().mean(1)
                            for c in classes], 1)

    def forward(self, x, task_id=None):
        end = min(self.num_classes, (self.seen_upto + 1) * self.classes_per_task)
        if end <= 0:
            raise RuntimeError('PEC has no observed classes')
        classes = (range(end) if task_id is None else
                   range(task_id * self.classes_per_task,
                         (task_id + 1) * self.classes_per_task))
        return -self.pec_error(x, classes)


class PEC(ContinualMethod):
    name = 'pec'

    def __init__(self, model, device, *args, num_classes=50, **kw):
        if not isinstance(model, PECNet):
            model = PECNet(num_classes, getattr(model, 'classes_per_task', 5))
        super().__init__(model, device, *args, **kw)

    def train_task(self, task_id, train_loader, val_loader, num_epochs=50,
                   patience=10, verbose=True):
        from tqdm import tqdm
        from training_policy import optimizer_for, EpochPolicy
        self.model.ensure_head(task_id)
        self.model.to(self.device)
        start = task_id * self.model.classes_per_task
        classes = range(start, start + self.model.classes_per_task)
        optimizers = {c: optimizer_for(self, self.model.students[c].parameters())
                      for c in classes}
        schedules = {c: EpochPolicy(self, opt, num_epochs, patience)
                     for c, opt in optimizers.items()}
        epoch_log = []
        best_loss, best_state, waited, epochs_run = float('inf'), None, 0, 0
        for _ in tqdm(range(num_epochs), desc=f'PEC task {task_id}', disable=not verbose):
            epochs_run += 1
            self.model.train()
            self.model.teacher.eval()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                if self.scenario == 'task_il':
                    y = y + start
                for c in y.unique().tolist():
                    if c not in optimizers:
                        raise ValueError('PEC received a label outside the current task')
                    sel = y == c
                    optimizers[c].zero_grad(set_to_none=True)
                    # Upstream loss: mean of the per-sample PEC error for class c.
                    loss = self.model.pec_error(x[sel], [c]).mean()
                    loss.backward()
                    optimizers[c].step()
            val_loss, _ = self.validate(val_loader, task_id)
            epoch_log.append(dict(epoch=epochs_run, validation_loss=val_loss))
            if val_loss < best_loss:
                best_loss, waited = val_loss, 0
                best_state = {c: copy.deepcopy(self.model.students[c].state_dict())
                              for c in classes}
            else:
                waited += 1
                if next(iter(schedules.values())).should_stop(epochs_run, waited):
                    break
            for schedule in schedules.values():
                schedule.step()
        if best_state:
            for c, state in best_state.items():
                self.model.students[c].load_state_dict(state)
        result = {'task_id': task_id, 'epochs_run': epochs_run,
                  'epoch_log': epoch_log, 'epoch_cap': num_epochs,
                  'cap_reached_while_improving': epochs_run == num_epochs and waited < patience,
                  'variant': 'PEC official mechanism (shared frozen teacher, independent '
                             'per-class students) adapted to the GTEP schedule',
                  'upstream': 'github.com/michalzajac-ml/pec@3c15633'}
        self.history.append(result)
        return result

    @torch.no_grad()
    def validate(self, loader, task_id):
        """Validation loss is the PEC error of the true class -- the quantity
        training minimises -- so early stopping tracks the training objective."""
        self.model.eval()
        loss_sum = correct = count = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            if self.scenario == 'task_il':
                scores = self.predict(x, task_id)
                target = y
            else:
                scores = self.predict(x, task_id)
                target = y
            loss_sum += -scores.gather(1, target[:, None]).sum().item()
            correct += scores.argmax(1).eq(target).sum().item()
            count += target.numel()
        return loss_sum / max(count, 1), correct / max(count, 1)
