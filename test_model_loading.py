import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from model_loading import load_model, model_resource


class TinyModel(torch.nn.Module):
    config_class = SimpleNamespace(from_pretrained=lambda *args, **kwargs: None)

    def __init__(self, config):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.register_buffer("positions", torch.arange(2), persistent=False)


class ModelLoadingTests(unittest.TestCase):
    def test_trained_checkpoint_values_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "model.safetensors")
            save_file({"weight": torch.tensor([3.0, 7.0])}, path)
            with patch("huggingface_hub.hf_hub_download", return_value=path):
                model = load_model(TinyModel, "test/model", local_files_only=True)
        torch.testing.assert_close(model.weight, torch.tensor([3.0, 7.0]))
        self.assertFalse(model.training)
        self.assertFalse(any(p.is_meta for p in list(model.parameters()) + list(model.buffers())))

    def test_missing_checkpoint_weights_are_rejected(self):
        with patch("huggingface_hub.hf_hub_download", return_value="unused"), patch("safetensors.torch.load_file", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                load_model(TinyModel, "test/model")

    def test_original_pytorch_checkpoint_loads_tensor_weights(self):
        from huggingface_hub.errors import EntryNotFoundError
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "pytorch_model.bin")
            torch.save({"weight": torch.tensor([4.0, 9.0])}, path)
            with patch("huggingface_hub.hf_hub_download", side_effect=[EntryNotFoundError("no safetensors"), path]):
                model = load_model(TinyModel, "test/model")
        torch.testing.assert_close(model.weight, torch.tensor([4.0, 9.0]))

    def test_legacy_position_buffers_must_match(self):
        with patch("huggingface_hub.hf_hub_download", return_value="unused"):
            with patch("safetensors.torch.load_file", return_value={"weight": torch.ones(2), "positions": torch.arange(2)}):
                model = load_model(TinyModel, "test/model")
                torch.testing.assert_close(model.weight, torch.ones(2))
            with patch("safetensors.torch.load_file", return_value={"weight": torch.ones(2), "positions": torch.tensor([5, 6])}):
                with self.assertRaisesRegex(RuntimeError, "Checkpoint buffer"):
                    load_model(TinyModel, "test/model")

    def test_existing_meta_context_gives_restart_instruction(self):
        class MetaModel(TinyModel):
            def __init__(self, config):
                super().__init__(config)
                self.weight = torch.nn.Parameter(torch.empty(2, device="meta"))
        with patch("huggingface_hub.hf_hub_download", return_value="unused"):
            with self.assertRaisesRegex(RuntimeError, "restart"):
                load_model(MetaModel, "test/model")

    def test_concurrent_calls_share_one_loaded_resource(self):
        calls = []
        @model_resource
        def factory():
            calls.append(1)
            time.sleep(.03)
            return object()
        with ThreadPoolExecutor(max_workers=4) as executor:
            resources = list(executor.map(lambda _: factory(), range(4)))
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(r is resources[0] for r in resources))
        factory.cache_clear()
        self.assertIsNot(factory(), resources[0])

    def test_different_models_do_not_initialize_concurrently(self):
        active, maximum = 0, 0
        guard = threading.Lock()
        def initialize():
            nonlocal active, maximum
            with guard:
                active += 1
                maximum = max(maximum, active)
            time.sleep(.03)
            with guard:
                active -= 1
            return object()
        audio, visual = model_resource(initialize), model_resource(initialize)
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(lambda fn: fn(), [audio, visual]))
        self.assertEqual(maximum, 1)

    def test_failed_load_can_be_retried(self):
        calls = []
        @model_resource
        def factory():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("temporary failure")
            return "loaded"
        with self.assertRaises(RuntimeError):
            factory()
        self.assertEqual(factory(), "loaded")


if __name__ == "__main__":
    unittest.main()
