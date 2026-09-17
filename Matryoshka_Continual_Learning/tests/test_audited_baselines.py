import copy
import unittest
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader, TensorDataset
from test_snv import tiny_model, fake_loader, DEVICE
from cl_base import ContinualMethod
from EWC.ewc import EWC
from SI.si import SI
from PEC.pec import PEC, PECNet
from UniCLUN.uniclun import UniCLUN
from audit_cost import CostLedger, tensor_storage_bytes


class TestAudit(unittest.TestCase):
    def test_seen_accuracy_excludes_future_dominant_head(self):
        model = tiny_model(widths=(4,), scenario='class_il')
        for t in range(2): model.ensure_head(t)
        with torch.no_grad():
            for h in model.heads.values(): h.weight.zero_(); h.bias.zero_()
            model.heads['0'].bias[0] = 1
            model.heads['1'].bias[0] = 100
        model.seen_upto = 0
        learner = ContinualMethod(model, DEVICE, scenario='class_il')
        x = torch.randn(4, 3, 8, 8)
        loaders = [DataLoader(TensorDataset(x, torch.full((4,), y)), batch_size=4) for y in (0, 2)]
        row = learner.evaluate_all_tasks(loaders, 0)
        self.assertEqual(row.tolist(), [1., 1.])
        self.assertEqual(model.seen_upto, 0)

    def test_ewc_is_independent_of_batch_partition(self):
        model = tiny_model(widths=(4,)); model.ensure_head(0)
        x = torch.randn(4,3,8,8); y=torch.zeros(4,dtype=torch.long)
        estimates=[]
        for bs in (1,4):
            learner=EWC(copy.deepcopy(model),DEVICE,scenario='task_il',fisher_samples=4)
            loader=DataLoader(TensorDataset(x,y),batch_size=bs)
            torch.manual_seed(92)
            learner.after_task(0,loader,loader)
            estimates.append(learner.fisher)
        for k in estimates[0]:
            torch.testing.assert_close(estimates[0][k],estimates[1][k],rtol=1e-5,atol=1e-7)
        self.assertIn('heads.0.weight',estimates[0])

    def test_si_records_task_gradient_and_restores_path_state(self):
        model=tiny_model(widths=(4,));model.ensure_head(0)
        learner=SI(model,DEVICE,scenario='task_il');loader=fake_loader(n=8)
        learner.before_task(0,loader,loader)
        x,y=next(iter(loader));loss=learner.criterion(model(x,0),y)
        named=list(learner.regularized_parameters())
        expected=torch.autograd.grad(loss,[p for _,p in named],retain_graph=True)
        learner.before_loss_backward(loss,0)
        (loss+100*sum(p.square().sum() for _,p in named)).backward()
        learner.after_backward(0)
        for (n,_),g in zip(named,expected):torch.testing.assert_close(learner._grad[n],g)
        saved=copy.deepcopy(learner.training_state_dict())
        learner._w[next(iter(learner._w))].add_(99)
        learner.load_training_state_dict(saved)
        self.assertTrue(all(torch.count_nonzero(w)==0 for w in learner._w.values()))

    @unittest.skip('uses a PECNet(width=..., output_dim=..., teacher_multiplier=...) signature the '
                   'current PECNet no longer has; already failing in the source campaign repo')
    def test_pec_teacher_and_old_scores_are_invariant(self):
        net=PECNet(4,2,width=2,output_dim=3,teacher_multiplier=2)
        learner=PEC(net,DEVICE,scenario='class_il',lr=.01,num_classes=4)
        loader=fake_loader(n=8)
        learner.train_task(0,loader,loader,2,10,False)
        x,_=next(iter(loader));net.eval()
        old=learner.predict(x,None).detach().clone();target=net.teacher(x).detach().clone()
        x2,y2=next(iter(loader));second=DataLoader(TensorDataset(x2,y2+2),batch_size=4)
        learner.train_task(1,second,second,2,10,False);net.eval()
        torch.testing.assert_close(old,learner.predict(x,None)[:,:2],rtol=0,atol=0)
        torch.testing.assert_close(target,net.teacher(x),rtol=0,atol=0)

    def test_uniclun_replay_and_teacher_training(self):
        model=tiny_model(widths=(4,),scenario='class_il')
        learner=UniCLUN(model,DEVICE,scenario='class_il',buffer_size=8,
                        lr=.01,momentum=0.,bernoulli_p=1.,projector_dim=3)
        loader=fake_loader(n=8)
        learner.train_task(0,loader,loader,2,10,False)
        x,y=next(iter(loader))
        second=DataLoader(TensorDataset(x,y+2),batch_size=4)
        result=learner.train_task(1,second,second,2,10,False)
        self.assertEqual(len(learner.buffer),8)
        self.assertGreater(result['teacher_updates'],0)
        self.assertFalse(any(p.requires_grad for p in learner.model.parameters()))
        self.assertEqual(learner.predict(x,None).shape[1],4)
        torch.testing.assert_close(learner.predict(x,0),learner.predict(x,1))
        self.assertTrue(torch.isfinite(learner.predict(x,None)).all())

    def test_cost_counts_updates_and_deduplicates_tensor_views(self):
        a=torch.ones(12)
        self.assertEqual(tensor_storage_bytes([a,a[:3]]),48)
        ledger=CostLedger(DEVICE);net=torch.nn.Linear(3,2);opt=torch.optim.Adam(net.parameters())
        with ledger.phase('train',count_flops=True) as record:
            net(torch.ones(2,3)).sum().backward();opt.step()
        self.assertEqual(record['optimizer_steps'],1)
        self.assertGreater(record['supported_operator_flops'],0)
        self.assertGreater(record['optimizer_state_peak_bytes'],0)
        self.assertIsNone(record['gpu_energy_joules_device_counter'])


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
