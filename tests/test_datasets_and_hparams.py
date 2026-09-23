"""The four benchmarks and the exported hyperparameters.

The dataset tests that need real images are skipped when the data directory
does not hold them, so the file is safe to run anywhere.
"""
import json
import os
import unittest

import numpy as np

import audited_gtep as G
from datasets import ContinualLearningBenchmark, build_transforms

DATA_ROOT = os.environ.get('GTEP_DATA_ROOT', './data')


def _has(*paths):
    return all(os.path.exists(os.path.join(DATA_ROOT, p)) for p in paths)


class TestGeometry(unittest.TestCase):
    def test_every_dataset_divides_into_its_tasks(self):
        expected = {'cifar20': (10, 2), 'cifar100': (50, 5),
                    'tinyimagenet': (100, 10), 'imagenet1k': (500, 50)}
        for dataset, want in expected.items():
            self.assertEqual(G.geometry(dataset), want, dataset)

    def test_task_count_is_five_for_cifar20_and_ten_elsewhere(self):
        self.assertEqual(G.task_count('cifar20'), 5)
        for dataset in ('cifar100', 'tinyimagenet', 'imagenet1k'):
            self.assertEqual(G.task_count(dataset), 10)

    def test_unknown_dataset_is_rejected(self):
        with self.assertRaises(ValueError):
            G.geometry('cifar42')

    def test_classes_per_half_times_tasks_is_the_half(self):
        for dataset in G.DATASET_CLASSES:
            half, per_task = G.geometry(dataset)
            self.assertEqual(half * 2, G.DATASET_CLASSES[dataset])
            self.assertEqual(per_task * G.task_count(dataset), half)


class TestTransforms(unittest.TestCase):
    def test_cifar20_uses_the_cifar_pipeline(self):
        for train in (True, False):
            a = build_transforms('cifar20', train)
            b = build_transforms('cifar100', train)
            self.assertEqual(str(a), str(b))


@unittest.skipUnless(_has('cifar-100-python'), 'CIFAR-100 not downloaded')
class TestCIFAR20(unittest.TestCase):
    def test_coarse_labels_are_the_twenty_superclasses(self):
        from datasets import CIFAR20
        pool = CIFAR20(DATA_ROOT, train=True, download=False)
        self.assertEqual(len(pool), 50000)
        self.assertEqual(sorted(set(pool.targets)), list(range(20)))

    def test_images_match_cifar100(self):
        from torchvision import datasets as tv
        from datasets import CIFAR20
        coarse = CIFAR20(DATA_ROOT, train=True, download=False)
        fine = tv.CIFAR100(DATA_ROOT, train=True, download=False)
        self.assertTrue(np.array_equal(coarse.data, fine.data))

    def test_halves_are_disjoint_and_split_seventy_ten_twenty(self):
        halves = []
        for half in (1, 2):
            b = ContinualLearningBenchmark('cifar20', 5, DATA_ROOT, 42, 'class_il', 0,
                                           gtep_half=half, split_seed=1234)
            self.assertEqual(b.num_classes, 10)
            self.assertEqual(b.classes_per_task, 2)
            halves.append(set(b.class_order.tolist()))
            train, val, test = (len(l.dataset) for l in b.get_task_data(0, 8))
            total = train + val + test
            self.assertAlmostEqual(train / total, 0.70, places=2)
            self.assertAlmostEqual(val / total, 0.10, places=2)
            self.assertAlmostEqual(test / total, 0.20, places=2)
        self.assertEqual(halves[0] & halves[1], set())
        self.assertEqual(halves[0] | halves[1], set(range(20)))

    def test_class_il_labels_are_global_and_task_il_labels_are_local(self):
        for scenario, first in (('class_il', 2), ('task_il', 0)):
            b = ContinualLearningBenchmark('cifar20', 5, DATA_ROOT, 42, scenario, 0,
                                           gtep_half=1, split_seed=1234)
            self.assertEqual(min(b.get_class_mapping(1).values()), first)


@unittest.skipUnless(_has('tiny-imagenet-200'), 'TinyImageNet not downloaded')
class TestTinyImageNet(unittest.TestCase):
    def test_tasks_are_ten_classes_of_64px_images(self):
        b = ContinualLearningBenchmark('tinyimagenet', 10, DATA_ROOT, 42, 'class_il', 0,
                                       gtep_half=1, split_seed=1234)
        self.assertEqual((b.num_classes, b.classes_per_task), (100, 10))
        images, _ = next(iter(b.get_task_data(0, 4)[0]))
        self.assertEqual(tuple(images.shape[1:]), (3, 64, 64))


class TestImageNetMessage(unittest.TestCase):
    def test_missing_imagenet_explains_where_to_put_it(self):
        b = ContinualLearningBenchmark('imagenet1k', 10, '/nonexistent', 42, 'class_il', 0,
                                       gtep_half=1, split_seed=1234)
        with self.assertRaises(FileNotFoundError) as caught:
            b.get_task_data(0, 4)
        self.assertIn('imagenet', str(caught.exception).lower())


class TestHyperparameters(unittest.TestCase):
    def test_every_shipped_method_has_a_winner(self):
        from hyperparameters import all_entries
        entries = all_entries()
        for method, scenarios in [('snv', ('class_il', 'task_il')),
                                  ('sgd', ('class_il', 'task_il')),
                                  ('joint', ('class_il', 'task_il')),
                                  ('ewc', ('class_il', 'task_il')),
                                  ('si', ('class_il', 'task_il')),
                                  ('lwf', ('class_il', 'task_il')),
                                  ('nispa', ('class_il', 'task_il')),
                                  ('spacenet', ('class_il', 'task_il')),
                                  ('uniclun', ('class_il', 'task_il')),
                                  ('wsn', ('task_il',)),      # TIL only
                                  ('pec', ('class_il',))]:     # CIL only
            for scenario in scenarios:
                self.assertIn(f'{method}/{scenario}', entries)

    def test_configs_only_use_keys_the_search_space_defines(self):
        from hyperparameters import all_entries
        allowed_extra = {'truncation', 'task_local', 'frozen_norm_eval', 'bn_recal',
                         'bn_recal_mode', 'adaptive', 'adaptive_rule', 'adaptive_coverage',
                         'routing', 'rot_aux'}
        for key, record in all_entries().items():
            method = record['block']
            space = G.SPACE.get('snv' if method == 'snv_adaptive' else method)
            if space is None:
                continue
            unknown = set(record['config']) - set(space) - allowed_extra
            self.assertEqual(unknown, set(), f'{key}: unexpected keys {unknown}')

    def test_snv_routing_differs_by_scenario(self):
        from hyperparameters import best_config
        self.assertEqual(best_config('snv', 'class_il')['routing'], 'rot_energy_z')
        self.assertEqual(best_config('snv', 'task_il')['routing'], 'maxprob')

    def test_json_and_markdown_agree_on_snv(self):
        import pathlib
        md = (pathlib.Path(__file__).resolve().parent.parent
              / 'hyperparameters' / 'cifar100_best.md').read_text()
        from hyperparameters import entry
        acc = entry('snv', 'class_il')['clean_eval_D_E']['ACC']
        self.assertIn(f'{acc:.4f}', md)

    def test_best_config_is_json_serialisable_for_the_worker(self):
        from hyperparameters import best_config
        json.loads(json.dumps(best_config('snv', 'class_il')))


if __name__ == '__main__':
    unittest.main()
