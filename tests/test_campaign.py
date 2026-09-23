"""The CIFAR-100 campaign: every method, both scenarios, the protocol settings.

These tests read the campaign module and the shipped results; they never train.
"""
import importlib.util
import json
import os
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ.setdefault('GTEP_PROTOCOL', 'legacy')


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RC = _load('run_campaign_under_test', 'campaign/run_campaign.py')
SAR = _load('snv_adaptive_run_under_test', 'snv_adaptive_run.py')
RESULTS = ROOT / 'results' / 'cifar100'


class TestCoverage(unittest.TestCase):
    def test_campaign_runs_every_method_the_registry_ships(self):
        import baselines
        self.assertEqual(set(RC.METHODS), set(baselines.ALL_METHODS))

    def test_every_method_has_a_search_space(self):
        import audited_gtep as G
        for method in RC.METHODS:
            self.assertIn(method, G.SPACE, method)
            self.assertIn('lr', G.SPACE[method], method)

    def test_scenarios_respect_the_two_restricted_methods(self):
        self.assertEqual(RC.scenarios('pec'), ['class_il'])    # CIL only
        self.assertEqual(RC.scenarios('wsn'), ['task_il'])     # needs the task id
        for method in RC.METHODS:
            if method not in ('pec', 'wsn'):
                self.assertEqual(RC.scenarios(method), ['class_il', 'task_il'], method)

    def test_snv_is_in_the_campaign_and_goes_through_its_own_worker(self):
        self.assertIn('snv', RC.METHODS)
        self.assertEqual(RC.scenarios('snv'), ['class_il', 'task_il'])
        self.assertTrue((ROOT / 'snv_adaptive_run.py').exists())
        self.assertTrue((ROOT / 'SNV' / 'snv_adaptive.py').exists())


class TestProtocolSettings(unittest.TestCase):
    def test_the_protocol_constants_are_the_reported_ones(self):
        self.assertEqual(RC.SEEDS, (42, 43, 44))
        self.assertEqual(RC.ROUNDS, 30)
        self.assertEqual(RC.EPOCHS, 200)
        self.assertEqual(RC.PATIENCE, 15)
        self.assertEqual(RC.TASKS, 10)
        self.assertEqual(RC.SAMPLE_SEED, 7)

    def test_the_legacy_protocol_is_the_default(self):
        self.assertEqual(os.environ['GTEP_PROTOCOL'], 'legacy')

    def test_draws_are_deterministic(self):
        for method in RC.METHODS:
            self.assertEqual(RC.configs_for(method, 'class_il'),
                             RC.configs_for(method, 'class_il'), method)

    def test_baselines_keep_all_thirty_draws(self):
        for method in RC.METHODS:
            if method == 'snv':
                continue
            indices = [i for i, _ in RC.configs_for(method, 'class_il')]
            self.assertEqual(indices, list(range(RC.ROUNDS)), method)


class TestSNVBlock(unittest.TestCase):
    def test_draws_below_the_lr_floor_are_dropped_and_indices_kept(self):
        for scenario in ('class_il', 'task_il'):
            indexed = RC.configs_for('snv', scenario)
            self.assertEqual(len(indexed), 23, scenario)
            self.assertEqual([i for i, _ in indexed],
                             [0, 1, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20,
                              21, 23, 24, 25, 26, 27, 28, 29], scenario)
            for _, config in indexed:
                self.assertGreaterEqual(config['lr'], RC.SNV_MIN_LR, scenario)

    def test_routing_differs_by_scenario_and_the_rest_does_not(self):
        cil = dict(RC.configs_for('snv', 'class_il'))[14]
        til = dict(RC.configs_for('snv', 'task_il'))[14]
        self.assertEqual((cil['routing'], cil['rot_aux']), ('rot_energy_z', 1.0))
        self.assertEqual((til['routing'], til['rot_aux']), ('maxprob', 0.0))
        shared = {k: v for k, v in cil.items() if k not in ('routing', 'rot_aux')}
        self.assertEqual(shared, {k: v for k, v in til.items()
                                  if k not in ('routing', 'rot_aux')})

    def test_every_switch_reaches_the_snv_constructor(self):
        """A key the runner does not read would be silently ignored."""
        handled = set(SAR.VARIANT_KEYS) | {'lr', 'truncation', 'max_permutations'}
        for scenario in ('class_il', 'task_il'):
            for _, config in RC.configs_for('snv', scenario):
                self.assertEqual(set(config) - handled, set(), scenario)

    def test_the_selected_configuration_is_one_of_the_searched_draws(self):
        from hyperparameters import best_config
        for scenario in ('class_il', 'task_il'):
            drawn = [c for _, c in RC.configs_for('snv', scenario)]
            self.assertIn(best_config('snv', scenario), drawn, scenario)


@unittest.skipUnless(RESULTS.exists(), 'no shipped results')
class TestResultsMatchTheCampaign(unittest.TestCase):
    def test_every_method_scenario_has_three_clean_runs(self):
        runs = {p.name for p in (RESULTS / 'runs').iterdir() if p.is_dir()}
        for method in RC.METHODS:
            block = 'snv_adaptive' if method == 'snv' else method
            for scenario in RC.scenarios(method):
                found = [r for r in runs
                         if r.startswith(f'{block}_{scenario}_clean_eval_s')]
                self.assertEqual(len(found), len(RC.SEEDS), f'{method}/{scenario}')

    def test_no_run_belongs_to_a_method_the_campaign_no_longer_has(self):
        blocks = {'snv_adaptive' if m == 'snv' else m for m in RC.METHODS}
        for run in (RESULTS / 'runs').iterdir():
            owner = run.name.split('_class_il')[0].split('_task_il')[0]
            self.assertIn(owner, blocks, run.name)

    def test_every_method_scenario_has_an_exported_winner(self):
        from hyperparameters import all_entries
        entries = all_entries()
        for method in RC.METHODS:
            for scenario in RC.scenarios(method):
                self.assertIn(f'{method}/{scenario}', entries)

    def test_the_summary_table_covers_the_same_runs(self):
        import csv
        with open(RESULTS / 'metrics_summary.csv') as f:
            rows = list(csv.DictReader(f))
        table = {(r['method'], r['scenario']) for r in rows}
        for method in RC.METHODS:
            for scenario in RC.scenarios(method):
                self.assertIn((method, scenario), table)
        for row in rows:
            self.assertEqual(int(row['seeds']), len(RC.SEEDS), row['method'])


if __name__ == '__main__':
    unittest.main()
