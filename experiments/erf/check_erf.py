"""Check ERF math and model extraction without ImageNet or pretrained downloads.

Run: .venv/Scripts/python.exe experiments/erf/check_erf.py
Random-weight models used here are test fixtures, not experimental results.
"""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch import nn
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
from torchvision.models import resnet50, vit_b_16
from torchvision.models.vision_transformer import VisionTransformer

import erf
from make_erf_notebook import build_notebook


class ERFChecks(unittest.TestCase):
    def test_metrics_known_distributions_and_scaling(self):
        uniform = np.ones((10, 10))
        point = np.zeros((10, 10))
        point[5, 5] = 1
        self.assertAlmostEqual(erf.erf_metrics(uniform)["area90_fraction"], 0.9)
        self.assertAlmostEqual(erf.erf_metrics(point)["area90_fraction"], 0.01)
        self.assertEqual(erf.erf_metrics(point)["dispersion_normalized"], 0)
        raw = np.random.default_rng(42).random((12, 15))
        a, b = erf.erf_metrics(raw), erf.erf_metrics(raw * 13)
        for name in a:
            if name != "raw_gradient_sum":
                self.assertAlmostEqual(a[name], b[name])
        self.assertLessEqual(a["area50_fraction"], a["area90_fraction"])

    def test_invalid_maps_fail(self):
        for raw in [np.zeros((2, 2)), np.ones(4), np.full((2, 2), np.nan), -np.ones((2, 2))]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                erf.erf_metrics(raw)

    def test_gradient_rgb_chain_rule_and_exact_spatial_support(self):
        # A 1x1 convolution makes the ERF exactly one pixel. Finite differences
        # below perturb RGB, before normalization, rather than normalized x.
        model = nn.Module()
        model.conv1 = nn.Conv2d(3, 1, 1, bias=False)
        model.conv1.weight.data[:] = torch.tensor([1., 2., 3.]).reshape(1, 3, 1, 1)
        for name in ["bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"]:
            setattr(model, name, nn.Identity())
        model.requires_grad_(False)
        x = torch.ones(1, 3, 224, 224)
        std = [2., 4., 8.]
        with torch.no_grad():  # compute_erf must re-enable input gradients.
            maps = erf.compute_erf(model, "resnet50", x, std)
        raw = maps["layer1"]["raw"]
        self.assertEqual(np.count_nonzero(raw), 1)
        self.assertAlmostEqual(float(raw[112, 112]), 16.5)
        eps = 0.01
        total = 0
        for channel in range(3):
            plus, minus = x.clone(), x.clone()
            plus[0, channel, 112, 112] += eps / std[channel]
            minus[0, channel, 112, 112] -= eps / std[channel]
            zp = erf.center_features(model, "resnet50", plus)[0]["layer1"].square().mean()
            zm = erf.center_features(model, "resnet50", minus)[0]["layer1"].square().mean()
            total += abs(float((zp - zm) / (2 * eps)))
        self.assertAlmostEqual(float(raw[112, 112]), total, places=2)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        with torch.inference_mode(), self.assertRaises(RuntimeError):
            erf.compute_erf(model, "resnet50", x, std)

    def test_real_model_paths_match_forward_hooks_and_backpropagate(self):
        torch.manual_seed(42)
        for name, factory in [("resnet50", resnet50), ("vit_b_16", vit_b_16)]:
            with self.subTest(model=name):
                model = factory(weights=None).eval().requires_grad_(False)
                captured = {}
                handles = []
                modules = ([getattr(model, layer) for layer in erf.LAYERS[name]]
                           if name == "resnet50"
                           else [model.encoder.layers[index] for index in [2, 5, 8, 11]])
                for layer, module in zip(erf.LAYERS[name], modules):
                    def capture(_module, _inputs, output, layer=layer):
                        captured[layer] = output.detach().clone()
                    handles.append(module.register_forward_hook(capture))
                x = torch.randn(1, 3, 224, 224)
                with torch.no_grad():
                    model(x)
                for handle in handles:
                    handle.remove()
                with torch.no_grad():
                    vectors, positions = erf.center_features(model, name, x)
                for layer, vector in vectors.items():
                    if name == "resnet50":
                        position = positions[layer]
                        expected = captured[layer][0, :, position['grid_row'], position['grid_col']]
                    else:
                        self.assertEqual(positions[layer]["token_index"], 106)
                        expected = captured[layer][0, 106, :]
                    torch.testing.assert_close(vector, expected)
                del captured, vectors
                maps = erf.compute_erf(model, name, x, [0.229, 0.224, 0.225])
                self.assertEqual(list(maps), erf.LAYERS[name])
                for result in maps.values():
                    raw = result["raw"]
                    self.assertEqual(raw.shape, (224, 224))
                    self.assertTrue(np.isfinite(raw).all())
                    self.assertGreater(raw.sum(), 0)
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                del model, maps

    def test_loader_indices_missing_ids_and_wrong_labels(self):
        from io import BytesIO
        buffer = BytesIO()
        Image.new("RGB", (8, 8), "blue").save(buffer, format="JPEG")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for index in range(14):
                pq.write_table(pa.Table.from_pylist([{
                    "image": {"path": f"image_{index}.JPEG", "bytes": buffer.getvalue()},
                    "label": index,
                }]), directory / f"validation-{index:05d}-of-00014.parquet")
            records = [{"dataset_index": i, "image_id": f"image_{i}.JPEG", "label": i}
                       for i in [3, 12]]
            loaded = list(erf.load_images(directory, records))
            self.assertEqual([row[0] for row in loaded], records)
            wrong = [{**records[0], "label": 999}]
            with self.assertRaises(ValueError):
                list(erf.load_images(directory, wrong))
            with self.assertRaises(ValueError):
                list(erf.load_images(directory, [{**records[0], "image_id": "missing"}]))

    def test_committed_manifest_and_standalone_notebook(self):
        selected = erf.select_records(ROOT, 10, 42)
        self.assertEqual(selected, erf.select_records(ROOT, 10, 42))
        self.assertEqual(len(selected), 10)
        notebook = json.loads((ROOT / "notebooks/04_erf.ipynb").read_text("utf-8"))
        self.assertEqual(notebook, build_notebook())
        namespace = {}
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                compile(source, f"cell-{index}", "exec")
                self.assertFalse(cell["outputs"])
                if "erf-source" in cell["metadata"].get("tags", []):
                    exec(source, namespace)
        self.assertEqual(namespace["erf_metrics"](np.ones((10, 10)))["area90_fraction"], 0.9)

    def test_run_writes_complete_consistent_artifacts(self):
        # Small random networks exercise the entire runner, with no downloads.
        # Their outputs stay in a temporary directory, outside research results.
        def small_cnn(weights):
            model = nn.Module()
            model.conv1 = nn.Conv2d(3, 4, 1)
            for name in ["bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"]:
                setattr(model, name, nn.Identity())
            return model

        def small_vit(weights):
            return VisionTransformer(image_size=224, patch_size=16, num_layers=12,
                                     num_heads=2, hidden_dim=8, mlp_dim=16)

        class TestWeights:
            def transforms(self):
                return erf.ResNet50_Weights.IMAGENET1K_V2.transforms()

        from io import BytesIO
        buffer = BytesIO()
        Image.new("RGB", (240, 240), "blue").save(buffer, format="JPEG")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for index in range(14):
                pq.write_table(pa.Table.from_pylist([{
                    "image": {"path": f"test_{index}.JPEG", "bytes": buffer.getvalue()},
                    "label": index,
                }]), directory / f"validation-{index:05d}-of-00014.parquet")
            records = [{"dataset_index": 0, "image_id": "test_0.JPEG", "label": 0}]
            output = directory / "synthetic_run"
            specs = {"resnet50": (small_cnn, TestWeights()),
                     "vit_b_16": (small_vit, TestWeights())}
            with patch.dict(erf.MODEL_SPECS, specs, clear=True):
                metrics, summary, previews = erf.run_experiment(
                    records, directory, output, torch.device("cpu"), num_previews=1,
                )
                self.assertEqual(len(metrics), 8)
                self.assertEqual(len(summary), 8)
                self.assertEqual(len(previews), 1)
                metadata = json.loads((output / "run.json").read_text("utf-8"))
                self.assertEqual(metadata["status"], "complete")
                self.assertEqual(len(list((output / "maps").glob("*.npz"))), 2)
                self.assertEqual(len(list((output / "figures").glob("*.png"))), 2)
                with np.load(next((output / "maps").glob("resnet50*.npz"))) as saved:
                    self.assertEqual(set(saved.files), set(erf.LAYERS["resnet50"]))
                with self.assertRaises(FileExistsError):
                    erf.run_experiment(records, directory, output, torch.device("cpu"))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
