"""Load the application's single-file checkpoints without meta initialization."""
import threading
from functools import lru_cache, wraps

_MODEL_LOAD_LOCK = threading.RLock()


def model_resource(factory):
    """Serialize cache misses across models and avoid duplicate concurrent loads."""
    cached = lru_cache(maxsize=1)(factory)

    @wraps(factory)
    def resource(*args, **kwargs):
        with _MODEL_LOAD_LOCK:
            return cached(*args, **kwargs)

    def clear():
        with _MODEL_LOAD_LOCK:
            cached.cache_clear()

    resource.cache_clear = clear
    return resource


def load_model(model_type, model_id, device="cpu", *, local_files_only=False):
    """Materialize trained weights on CPU, then move the complete model.

    OWL-ViT and CLAP supply single-file checkpoints.
    Strict loading rejects missing weights instead of running random parameters.
    No to_empty() or process-wide Transformers initialization patches are used.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    from safetensors.torch import load_file

    with _MODEL_LOAD_LOCK:
        config = model_type.config_class.from_pretrained(model_id, local_files_only=local_files_only)
        try:
            checkpoint = hf_hub_download(model_id, "model.safetensors", local_files_only=local_files_only)
            safe_format = True
        except EntryNotFoundError:
            # CLAP's original release has a .bin checkpoint. Restrict loading
            # to tensor weights; do not deserialize arbitrary Python objects.
            checkpoint = hf_hub_download(model_id, "pytorch_model.bin", local_files_only=local_files_only)
            safe_format = False
        with torch.device("cpu"):
            model = model_type(config)
        tensors = list(model.named_parameters()) + list(model.named_buffers())
        if any(tensor.is_meta for _, tensor in tensors):
            raise RuntimeError(
                "Model initialization is still affected by an old meta-tensor context. "
                "Stop Streamlit with Ctrl+C and restart it to clear the old process state."
            )
        weights = (load_file(checkpoint, device="cpu") if safe_format
                   else torch.load(checkpoint, map_location="cpu", weights_only=True))
        # Older checkpoints saved deterministic position IDs that newer model
        # classes register as non-persistent buffers. Validate before omitting.
        persistent_keys = model.state_dict().keys()
        for name, buffer in model.named_buffers():
            if name in weights and name not in persistent_keys:
                if not torch.equal(weights[name], buffer):
                    raise RuntimeError(f"Checkpoint buffer {name} does not match this model's configuration.")
                del weights[name]
        model.load_state_dict(weights, strict=True)
        del weights
        return model.to(device).eval()
