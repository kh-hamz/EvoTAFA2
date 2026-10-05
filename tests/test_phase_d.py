"""Known-answer Phase D numerical and contract regressions."""
import tempfile
import unittest
from pathlib import Path
import torch
from torch.utils.data import TensorDataset
from edgefl.learning.aggregation import reduce_updates
from edgefl.learning.models import initialized, configure
from edgefl.learning.training import train_epochs, adam
from edgefl.learning.metrics import summarize, magnitude, weight_change, evaluate
from edgefl.learning import checkpoints


class NumericalTests(unittest.TestCase):
    def setUp(self):
        self.values = {"device": "cpu", "threads": 1, "learning_rate": .02, "batch_size": 8}
        configure(self.values)

    def test_fedavg_registered_counts_and_even_median(self):
        updates = {"a": {"w": torch.tensor([1., 9.])}, "b": {"w": torch.tensor([5., 1.])}}
        delta, weights, _ = reduce_updates(updates, {"a": 1, "b": 3}, "fedavg")
        torch.testing.assert_close(delta["w"], torch.tensor([4., 3.]))
        self.assertEqual(weights, {"a": .25, "b": .75})
        result, weights, _ = reduce_updates(updates, {}, "median")
        torch.testing.assert_close(result["w"], torch.tensor([3., 5.]))
        self.assertIsNone(weights)

    def test_trimmed_mean_and_insufficient_clients(self):
        updates = {str(i): {"w": torch.tensor([v])} for i, v in enumerate((-100., 1., 2., 3., 100.))}
        result, _, _ = reduce_updates(updates, {}, "trimmed_mean")
        self.assertEqual(float(result["w"]), 2.)
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            reduce_updates({"a": updates["0"]}, {}, "trimmed_mean")

    def test_fltrust_direction_norm_and_zero_cases(self):
        root = {"w": torch.tensor([2., 0.])}
        updates = {"aligned": {"w": torch.tensor([20., 0.])}, "opposite": {"w": torch.tensor([-1., 0.])},
                   "orthogonal": {"w": torch.tensor([0., 1.])}, "zero": {"w": torch.zeros(2)}}
        result, weights, scales = reduce_updates(updates, {}, "fltrust", root=root)
        torch.testing.assert_close(result["w"], root["w"])
        self.assertEqual(weights["aligned"], 1)
        self.assertEqual(scales["aligned"], .1)
        for update, root_value in ((updates, {"w": torch.zeros(2)}), ({"bad": updates["opposite"]}, root)):
            with self.assertRaises(ValueError):
                reduce_updates(update, {}, "fltrust", root=root_value)

    def test_fedprox_zero_equivalence_and_proximal_gradient(self):
        data = TensorDataset(torch.tensor([[-1.], [1.]] * 4), torch.tensor([0, 1] * 4))
        models = [initialized(1, 2, 11, "logistic") for _ in range(2)]
        parent = checkpoints.cpu_state(models[0])
        for model, mu in zip(models, (0., 0.)):
            train_epochs(model, data, adam(model, self.values), self.values, 11, "c", 1, 2, parent, mu)
        for a, b in zip(models[0].parameters(), models[1].parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        x = torch.tensor([3., 5.], requires_grad=True)
        reference = torch.tensor([1., 2.])
        (.4 / 2 * (x - reference).square().sum()).backward()
        torch.testing.assert_close(x.grad, torch.tensor([.8, 1.2]))

    def test_tiny_mlp_and_logistic_learn(self):
        data = TensorDataset(torch.tensor([[-2.], [-1.], [1.], [2.]] * 4), torch.tensor([0, 0, 1, 1] * 4))
        for family in ("mlp", "logistic"):
            model = initialized(1, 2, 11, family)
            before = evaluate(model, data, 2, 0, 8, "cpu", "local_train")
            train_epochs(model, data, adam(model, self.values), self.values, 11, "c", 1, 20)
            after = evaluate(model, data, 2, 0, 8, "cpu", "local_train")
            self.assertLess(after["loss"], before["loss"])
            self.assertGreater(after["macro_f1"], .9)

    def test_metrics_support_and_weight_turnover(self):
        report = summarize(torch.tensor([[8, 2], [1, 9]]), 10)
        self.assertEqual(report["loss"], .5)
        self.assertAlmostEqual(report["macro_f1"], (16 / 19 + 18 / 21) / 2)
        self.assertEqual(weight_change({"a": 1.}, {"b": 1.})["distance"], 1)
        self.assertTrue(weight_change({"a": 1.}, {"b": 1.})["participation_changed"])
        self.assertIsNone(weight_change(None, {})["distance"])
        self.assertIsNone(magnitude({"w": torch.ones(1)}, {"w": torch.zeros(1)})["relative"])
        model = initialized(1, 2, 11, "logistic")
        no_benign = TensorDataset(torch.ones((2, 1)), torch.ones(2, dtype=torch.int64))
        self.assertIsNone(evaluate(model, no_benign, 2, 0, 8, "cpu", "local_validation")["benign_fpr"])

    def test_initialization_and_checkpoint_tamper(self):
        a, b = initialized(3, 2, 11), initialized(3, 2, 11)
        for x, y in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.pt"
            digest = checkpoints.save(path, {"model": a.state_dict()})
            checkpoints.validate_state(checkpoints.load(path, digest)["model"], b.state_dict())
            with self.assertRaises(ValueError):
                checkpoints.save(path, {})
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash"):
                checkpoints.load(path, digest)

    def test_invalid_parameter_keys_shapes_and_values(self):
        template = {"w": torch.zeros(2)}
        for invalid in ({"wrong": torch.zeros(2)}, {"w": torch.zeros(3)}, {"w": torch.tensor([float("nan"), 0.])}):
            with self.assertRaises(ValueError):
                checkpoints.validate_state(invalid, template)

    def test_actual_proximal_training_gradient(self):
        model = torch.nn.Sequential(torch.nn.Linear(1, 2))
        with torch.no_grad():
            model[0].weight.copy_(torch.tensor([[1.], [-1.]]))
            model[0].bias.zero_()
        parent = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
        data = TensorDataset(torch.tensor([[2.]]), torch.tensor([0]))
        optimizer = adam(model, self.values)
        from unittest.mock import patch
        with patch.object(optimizer, "step"):
            train_epochs(model, data, optimizer, self.values, 11, "c", 1, 1, parent, .4)
        probability = 1 / (1 + __import__("math").exp(-4))
        expected = torch.tensor([[(probability - 1) * 2 + .4], [(1 - probability) * 2 - .4]])
        torch.testing.assert_close(model[0].weight.grad, expected)

    def test_single_client_matches_independent_adam_reference(self):
        from torch.utils.data import DataLoader
        from edgefl.learning.training import LocalTrainer
        from edgefl.contracts.records import Compatibility, ClientManifest, PartitionRef, Partition
        from edgefl.contracts.interfaces import TrainingRequest
        from edgefl.reproducibility import derive_seed
        data = TensorDataset(torch.tensor([[-1.], [1.]] * 4), torch.tensor([0, 1] * 4))
        values = {**self.values, "local_epochs": 1}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            comp = Compatibility("a" * 64, "b" * 64, "c" * 64, "mlp-float32-v1", 1, 2)
            reference_model = initialized(1, 2, 11)
            initial = checkpoints.cpu_state(reference_model)
            parent_path = root / "parent.pt"
            checkpoints.save(parent_path, {"model": initial})
            ref = checkpoints.artifact(root, parent_path)
            client = ClientManifest("c", ref, PartitionRef(ref, Partition.LOCAL_TRAIN), PartitionRef(ref, Partition.LOCAL_VALIDATION), (), 11)
            trainer = LocalTrainer(root, root, {"c": data}, comp, values, 11)
            generator_seed = derive_seed(11, "minibatch", client="c", round_id=1)
            update = trainer.train(TrainingRequest(client, 1, ref, comp, generator_seed))
            optimizer = torch.optim.Adam(reference_model.parameters(), lr=values["learning_rate"], foreach=False)
            loader = DataLoader(data, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(generator_seed))
            for x, y in loader:
                optimizer.zero_grad()
                torch.nn.functional.cross_entropy(reference_model(x), y).backward()
                optimizer.step()
            delta = checkpoints.load(root / update.delta.path)["delta"]
            aggregated, _, _ = reduce_updates({"c": delta}, {"c": len(data)}, "fedavg")
            for key, expected in reference_model.state_dict().items():
                torch.testing.assert_close(initial[key] + aggregated[key], expected, rtol=0, atol=0)
