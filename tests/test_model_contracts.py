"""Synthetic invariants for the refactored public implementation, not performance tests."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

class ModelContracts(unittest.TestCase):
    """Small shape and freezing checks for the public model paths."""
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.c4 = load('public_c4', 'src/models/correction/c4.py')
        cls.modules = cls.c4.load_formal_data.__globals__['load_official_modules'](str(ROOT))
        cls.hsdt = cls.modules.p108

    def make_model(self):
        p = self.modules
        adj = np.eye(7, dtype=np.float32)
        eta = p.priority1.GraphWaveNetForecaster(34, adj, 8, 24, 2, 2, 0.0)
        multi = p.p104.GraphWaveNetMultistate(34, adj, 8, 24, 4, 2, 2, 0.0)
        model = self.c4.AlignedC4(self.c4.EtaContext(eta), self.c4.MultiContext(multi), 8, 24)
        for expert in (model.eta_expert, model.multi_expert):
            for param in expert.parameters():
                param.requires_grad = False
        return model.eval()

    def test_fusion_rule(self):
        weights = self.hsdt.fusion_weights(24, 'horizon_specialized_dual_task_gwn')
        np.testing.assert_array_equal(weights[:-1], np.full(23, .5))
        self.assertEqual(weights[-1], 1.0)

    def test_initial_c4_matches_hsdt(self):
        torch.manual_seed(42)
        model = self.make_model()
        with torch.no_grad():
            out = model(torch.randn(2, 24, 7, 34), 1.0)
        expected = .5 * (out.eta + out.multi_states[..., 0])
        expected[..., -1] = out.multi_states[..., -1, 0]
        torch.testing.assert_close(out.mean, expected)
        self.assertEqual(tuple(out.mean.shape), (2, 7, 24))
        self.assertEqual(tuple(out.multi_states.shape), (2, 7, 24, 4))
        self.assertTrue(torch.isfinite(out.log_sigma_z).all())

    def test_frozen_expert_batchnorm_does_not_drift(self):
        model = self.make_model()
        before = {k: v.clone() for k, v in model.eta_expert.state_dict().items() if 'running_' in k}
        model.train()
        self.c4.keep_frozen_experts_eval(model, True)
        model(torch.randn(2, 24, 7, 34), 1.0)
        after = model.eta_expert.state_dict()
        for key, value in before.items():
            torch.testing.assert_close(after[key], value, rtol=0, atol=0)

    def test_validation_scale_shape_guard(self):
        metrics = load('public_metrics', 'src/models/correction/shared/metrics.py')
        with self.assertRaises(ValueError):
            metrics.fit_global_sigma_scale(np.zeros((2, 7)), np.zeros((2, 7)), np.ones((2, 1)))

    def test_c4_gradients_leave_experts_frozen(self):
        model = self.make_model().train()
        self.c4.keep_frozen_experts_eval(model, True)
        output = model(torch.randn(2, 24, 7, 34), 1.0)
        loss = self.c4.gaussian_nll(torch.randn_like(output.mean), output.mean, output.log_sigma_z)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.eta_expert.parameters()))
        self.assertTrue(all(p.grad is None for p in model.multi_expert.parameters()))
        self.assertGreater(model.correction.net[-1].weight.grad.abs().sum().item(), 0)

    def test_varx_feature_and_target_order(self):
        from types import SimpleNamespace
        varx = load('public_varx', 'src/models/linear/varx.py')
        dataset = SimpleNamespace(x_scaled=np.arange(8*7*34).reshape(8, 7, 34),
                                  residual=np.arange(8*7).reshape(8, 7),
                                  tide=np.zeros((8, 7)), indices=[2, 3], window=2, horizon=2)
        x, y, tide = varx.design_matrix(dataset)
        self.assertEqual(x.shape, (2, 490))
        np.testing.assert_array_equal(y[0], dataset.residual[2:4].T.reshape(-1))
        self.assertEqual(tide.shape, (2, 7, 2))

    def test_circular_sampler_preserves_exact_length(self):
        control = load('public_controls', 'src/evaluation/ensemble_scale_controls.py')
        result = control.bootstrap_totals(np.ones((401, 2)), 168, 42)
        np.testing.assert_array_equal(result, np.full((5000, 2), 401.0))

if __name__ == '__main__':
    unittest.main()
