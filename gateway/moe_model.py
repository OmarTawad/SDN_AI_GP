"""Mixture-of-Experts model that combines frozen Autoencoder, CNN, and LSTM experts.

This module expects the project layout:
    autoencoder/
    dosdet/
    Neural_LSTM/

The helper utilities below attempt to locate each expert's ``model.py`` definition and
its trained ``.pth`` weights automatically. Update the configuration dictionaries where
indicated if the automatic search does not find the correct files.

The Mixture-of-Experts (MoE) consists of:
    * A gating network (Linear → ReLU → Linear → Softmax) producing mixture weights
    * Three frozen experts (autoencoder, CNN, LSTM) whose parameters never update
    * A weighted combination of expert logits for binary DoS detection

Run this module directly to see a dummy training loop using synthetic data.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None

import joblib
import torch
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Project paths and configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTOENCODER_ROOT = PROJECT_ROOT / "autoencoder"
CNN_ROOT = PROJECT_ROOT / "dosdet"
LSTM_ROOT = PROJECT_ROOT / "Neural_LSTM"


def _extend_sys_path(*roots: Path) -> None:
    """Ensure the provided roots (and their src folders) are importable."""
    for root in roots:
        if not root.exists():
            continue
        for candidate in {root, root / "src"}:
            candidate_str = str(candidate.resolve())
            if candidate.exists() and candidate_str not in sys.path:
                sys.path.append(candidate_str)


_extend_sys_path(AUTOENCODER_ROOT, CNN_ROOT, LSTM_ROOT)

if yaml is None:  # pragma: no cover - configuration requires PyYAML
    raise ImportError("PyYAML is required to configure the Mixture-of-Experts model.")

from Neural_LSTM.src.dos_detector.config import load_config
from Neural_LSTM.src.dos_detector.config.types import SupervisedModelConfig


# ---------------------------------------------------------------------------
# Expert metadata (dimensions, kwargs, transforms)
# ---------------------------------------------------------------------------

AUTO_MODEL_CONFIG_PATH = AUTOENCODER_ROOT / "data" / "artifacts" / "model_config.json"
AUTO_CLIP_BOUNDS_PATH = AUTOENCODER_ROOT / "data" / "artifacts" / "clip_bounds.json"

auto_model_config = json.loads(AUTO_MODEL_CONFIG_PATH.read_text())
auto_model_params = auto_model_config.get("model", {})
AUTO_FEATURE_NAMES: List[str] = list(auto_model_config.get("feature_names", []))
AUTO_MODEL_KWARGS: Dict[str, Any] = {
    "input_dim": int(auto_model_params.get("input_dim") or len(AUTO_FEATURE_NAMES)),
    "hidden_layers": [int(x) for x in auto_model_params.get("hidden", [128, 64, 32])],
    "latent_dim": int(auto_model_params.get("latent_dim", 16)),
    "activation": auto_model_params.get("activation", "ReLU"),
    "dropout": float(auto_model_params.get("dropout", 0.0)),
}

DOSDET_CONFIG_PATH = CNN_ROOT / "config.yaml"
with DOSDET_CONFIG_PATH.open("r", encoding="utf-8") as cfg_file:
    dos_config = yaml.safe_load(cfg_file)

dos_training = dos_config["training"]
dos_windowing = dos_config["windowing"]
CNN_CHANNELS = tuple(dos_training.get("channels", [64, 96]))
CNN_KERNEL = int(dos_training.get("kernel_size", 5))
CNN_DROPOUT = float(dos_training.get("dropout", 0.1))
CNN_MLP = tuple(dos_training.get("mlp_hidden", [256, 64]))
CNN_AUX_FAMILY = bool(dos_training.get("aux_family_head", True))
CNN_NUM_FAMILIES = int(dos_training.get("n_families", 6)) if "n_families" in dos_training else 6

cnn_meta = json.loads((CNN_ROOT / "artifacts_fast" / "feature_model_meta.json").read_text())
CNN_SEQ_IN_DIM = int(cnn_meta.get("seq_in_dim", 14))
CNN_STATIC_DIM = int(cnn_meta.get("static_dim", 40))
CNN_MICRO_BINS = int(cnn_meta.get("micro_bins", dos_windowing.get("micro_bins", 8)))

LSTM_CONFIG_PATH = LSTM_ROOT / "configs" / "config.yaml"
LSTM_CONFIG = load_config(LSTM_CONFIG_PATH)
LSTM_MODEL_CONFIG: SupervisedModelConfig = LSTM_CONFIG.model.supervised
LSTM_NUM_TYPES = len(LSTM_CONFIG.labels.family_mapping)
LSTM_SCALER_PATH = LSTM_ROOT / "models" / "feature_scaler.joblib"
LSTM_SCALER = joblib.load(LSTM_SCALER_PATH)
LSTM_INPUT_DIM = int(getattr(LSTM_SCALER, "n_features_in_", 50))


def _auto_input_transform(sample: Any) -> Tensor:
    if isinstance(sample, dict) and "auto" in sample:
        return sample["auto"]
    if isinstance(sample, Tensor):
        return sample
    raise TypeError("Autoencoder expert expects a tensor under key 'auto'.")


def _cnn_input_transform(sample: Any) -> Tuple[Tensor, Tensor]:
    if isinstance(sample, dict):
        return sample["cnn_seq"], sample["cnn_static"]
    raise TypeError("CNN expert expects 'cnn_seq' and 'cnn_static' tensors in the feature dict.")


def _lstm_input_transform(sample: Any) -> Tensor:
    if isinstance(sample, dict) and "lstm_seq" in sample:
        return sample["lstm_seq"]
    if isinstance(sample, Tensor):
        return sample
    raise TypeError("LSTM expert expects a tensor under key 'lstm_seq'.")


def _lstm_output_transform(output: Any) -> Tensor:
    if hasattr(output, "file_logits"):
        logits = output.file_logits
    elif isinstance(output, dict) and "file_logits" in output:
        logits = output["file_logits"]
    else:
        raise TypeError("SequenceClassifier output missing 'file_logits'.")
    if isinstance(logits, Tensor):
        return logits
    raise TypeError("SequenceClassifier 'file_logits' is not a tensor.")


# ---------------------------------------------------------------------------
# Expert loading helpers
# ---------------------------------------------------------------------------

@dataclass
class ExpertSpec:
    """Configuration describing how to load and wrap a frozen expert."""

    name: str
    root: Path
    # Glob pattern used to locate the python file containing the expert (falls back to
    # ``model.py`` if present). Update if your model file is named differently.
    model_pattern: str = "model.py"
    # Optional: fully qualified attribute name to instantiate (class or builder).
    target_name: Optional[str] = None
    # Keyword arguments passed to the target when instantiating the expert.
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Glob pattern for the weight checkpoint. Update this to point at your .pth file.
    weight_pattern: str = "*.pth"  # TODO: Set to your actual checkpoint filename.
    # Optional transforms to adapt inputs/outputs per expert.
    input_transform: Optional[Callable[[Tensor], Tensor]] = None
    output_transform: Optional[Callable[[Any], Tensor]] = None


def _find_first(root: Path, pattern: str) -> Path:
    """Find the first file matching *pattern* within *root* (depth-first)."""
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Unable to locate '{pattern}' under {root}")
    return matches[0]


def _infer_dotted_module(module_path: Path) -> Optional[str]:
    package_parts: List[str] = []
    parent = module_path.parent
    while (parent / "__init__.py").exists():
        package_parts.insert(0, parent.name)
        parent = parent.parent
    if not package_parts:
        return None
    module_parts = package_parts + [module_path.stem]
    return ".".join(module_parts)


def _import_from_path(module_path: Path, module_name: str) -> ModuleType:
    """Import a module from an arbitrary filesystem path."""
    dotted = _infer_dotted_module(module_path)
    if dotted is not None:
        try:
            return importlib.import_module(dotted)
        except Exception:  # noqa: BLE001
            pass
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_target(module: ModuleType, target_name: Optional[str]) -> Callable[..., nn.Module]:
    """Pick the constructor callable from the imported module."""
    if target_name is not None:
        try:
            return getattr(module, target_name)
        except AttributeError as exc:
            raise AttributeError(
                f"Module '{module.__name__}' does not define '{target_name}'."
            ) from exc

    # Fallback: pick the first nn.Module subclass defined in the module.
    candidates = [
        obj
        for obj in module.__dict__.values()
        if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module
    ]
    if not candidates:
        raise ValueError(
            f"Could not find an nn.Module subclass inside module '{module.__name__}'. "
            "Provide 'target_name' in the ExpertSpec to select the correct class."
        )
    return candidates[0]


def _load_state_dict(module: nn.Module, weight_path: Path) -> None:
    """Load parameters into *module* from *weight_path* while staying on CPU."""
    state_dict = torch.load(weight_path, map_location="cpu")
    if isinstance(state_dict, dict):
        if all(isinstance(k, str) for k in state_dict.keys()):
            try:
                module.load_state_dict(state_dict)
                return
            except Exception:  # noqa: BLE001
                pass
        for key in ("state_dict", "model"):
            nested = state_dict.get(key)
            if isinstance(nested, dict):
                module.load_state_dict(nested)
                return
    raise RuntimeError(
        f"Checkpoint at '{weight_path}' is not a state_dict compatible with "
        f"{module.__class__.__name__}."
    )


class FrozenExpert(nn.Module):
    """Wrapper around a pre-trained expert that keeps its parameters frozen."""

    def __init__(
        self,
        name: str,
        module: nn.Module,
        input_transform: Optional[Callable[[Any], Any]] = None,
        output_transform: Optional[Callable[..., Tensor]] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.module = module
        self.input_transform = input_transform
        self.output_transform = output_transform
        self._freeze_params()

    def _freeze_params(self) -> None:
        self.module.eval()
        for param in self.module.parameters():
            param.requires_grad = False

    def forward(self, features: Any) -> Tensor:  # type: ignore[override]
        with torch.no_grad():
            x = self.input_transform(features) if self.input_transform else features
            if isinstance(x, dict):
                raw_output = self.module(**x)
            elif isinstance(x, (tuple, list)):
                raw_output = self.module(*x)
            else:
                raw_output = self.module(x)
        if self.output_transform:
            try:
                return self.output_transform(raw_output, x)
            except TypeError:
                return self.output_transform(raw_output)
        if isinstance(raw_output, dict):
            if "logits" in raw_output:
                return raw_output["logits"]
            raise ValueError(
                f"Expert '{self.name}' returned a dict without a 'logits' key. "
                "Provide an output_transform in ExpertSpec to extract the tensor."
            )
        if not isinstance(raw_output, Tensor):
            raise TypeError(
                f"Expert '{self.name}' returned unsupported type {type(raw_output)}. "
                "Provide an output_transform in ExpertSpec to convert it to a Tensor."
            )
        return raw_output

    def train(self, mode: bool = True) -> FrozenExpert:  # type: ignore[override]
        # Keep experts in eval mode regardless of the requested flag.
        self.module.eval()
        return self

    def to(self, *args: Any, **kwargs: Any) -> FrozenExpert:  # type: ignore[override]
        self.module.to(*args, **kwargs)
        return super().to(*args, **kwargs)


def load_frozen_expert(spec: ExpertSpec, device: Optional[torch.device] = None) -> FrozenExpert:
    """Build and freeze an expert according to *spec*."""
    model_path = _find_first(spec.root, spec.model_pattern)
    module_name = f"{spec.name}_module"
    module = _import_from_path(model_path, module_name)
    target_ctor = _resolve_target(module, spec.target_name)
    expert_module = target_ctor(**spec.model_kwargs)
    weight_path = _find_first(spec.root, spec.weight_pattern)
    _load_state_dict(expert_module, weight_path)
    expert_module.to(device or torch.device("cpu"))
    return FrozenExpert(
        name=spec.name,
        module=expert_module,
        input_transform=spec.input_transform,
        output_transform=spec.output_transform,
    )


# ---------------------------------------------------------------------------
# Gating network and Mixture-of-Experts container
# ---------------------------------------------------------------------------

class GatingNetwork(nn.Module):
    """Two-layer gating MLP that outputs a mixture distribution over experts."""

    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, features: Tensor) -> Tensor:  # type: ignore[override]
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        logits = self.fc2(self.act(self.fc1(features)))
        return self.softmax(logits)


class DosMoE(nn.Module):
    """Mixture-of-Experts head that keeps experts frozen and trains a gate."""

    def __init__(self, gating: GatingNetwork, experts: Iterable[FrozenExpert]) -> None:
        super().__init__()
        experts = list(experts)
        if len(experts) == 0:
            raise ValueError("At least one expert is required for the MoE.")
        self.gating = gating
        self.experts = nn.ModuleList(experts)

    def forward(self, features: Any) -> Tensor:  # type: ignore[override]
        if isinstance(features, dict):
            gating_input = features.get("gating")
            if gating_input is None:
                raise KeyError("Expected 'gating' key in feature dict for gating network input.")
        else:
            gating_input = features

        gate_weights = self.gating(gating_input)  # [B, E]
        expert_outputs: List[Tensor] = []
        for expert in self.experts:
            out = expert(features)
            if out.dim() == 1:
                out = out.unsqueeze(-1)
            elif out.dim() == 2 and out.size(1) != 1:
                # Ensure outputs broadcast correctly; collapse feature dims to 1.
                out = out.mean(dim=1, keepdim=True)
            expert_outputs.append(out)
        stacked = torch.stack(expert_outputs, dim=1)  # [B, E, 1]
        mixed = (stacked * gate_weights.unsqueeze(-1)).sum(dim=1)
        return mixed.squeeze(-1)

    def train(self, mode: bool = True) -> DosMoE:  # type: ignore[override]
        self.gating.train(mode)
        for expert in self.experts:
            expert.eval()
        self.training = mode
        return self

    def eval(self) -> DosMoE:  # type: ignore[override]
        return self.train(False)


# ---------------------------------------------------------------------------
# Example configuration and demo loop
# ---------------------------------------------------------------------------

# Update the specs below so they point at the correct model/weight files for your
# environment. Use ``target_name`` if the expert module exports multiple classes and you
# need to pick a specific one. ``model_kwargs`` should match the constructor signature of
# the expert so that the loaded state_dict fits.


# ----- expert specifications -------------------------------------------------

AUTOENCODER_SPEC = ExpertSpec(
    name="autoencoder",
    root=AUTOENCODER_ROOT,
    model_pattern="src/dae/model.py",
    target_name="FeedForwardAutoencoder",
    model_kwargs=AUTO_MODEL_KWARGS,
    weight_pattern="data/artifacts/model.pt",
    input_transform=_auto_input_transform,
    output_transform=lambda recon, original: -((original - recon) ** 2).mean(dim=1, keepdim=True),
)

CNN_MODEL_KWARGS: Dict[str, Any] = {
    "seq_in_dim": CNN_SEQ_IN_DIM,
    "static_dim": CNN_STATIC_DIM,
    "channels": CNN_CHANNELS,
    "k": CNN_KERNEL,
    "drop": CNN_DROPOUT,
    "mlp_hidden": CNN_MLP,
    "aux_family_head": CNN_AUX_FAMILY,
    "n_families": CNN_NUM_FAMILIES,
}

CNN_SPEC = ExpertSpec(
    name="cnn",
    root=CNN_ROOT,
    model_pattern="models/dws_cnn.py",
    target_name="FastDetector",
    model_kwargs=CNN_MODEL_KWARGS,
    weight_pattern="artifacts_fast/model_best.pt",
    input_transform=_cnn_input_transform,
    output_transform=None,
)

LSTM_MODEL_KWARGS: Dict[str, Any] = {
    "input_size": LSTM_INPUT_DIM,
    "num_attack_types": LSTM_NUM_TYPES,
    "config": LSTM_MODEL_CONFIG,
}

LSTM_SPEC = ExpertSpec(
    name="lstm",
    root=LSTM_ROOT,
    model_pattern="src/dos_detector/models/supervised.py",
    target_name="SequenceClassifier",
    model_kwargs=LSTM_MODEL_KWARGS,
    weight_pattern="models/supervised.pt",
    input_transform=_lstm_input_transform,
    output_transform=_lstm_output_transform,
)

print(
    "[MoE] Expert inputs: autoencoder=%dd, cnn_seq=%dx%d, cnn_static=%dd, lstm_seq=%dx%d"
    % (
        len(AUTO_FEATURE_NAMES),
        CNN_MICRO_BINS,
        CNN_SEQ_IN_DIM,
        CNN_STATIC_DIM,
        LSTM_CONFIG.windowing.sequence_length,
        LSTM_INPUT_DIM,
    )
)



def build_moe_model(
    specs: Iterable[ExpertSpec],
    gating_input_dim: int,
    gating_hidden_dim: int,
    device: Optional[torch.device] = None,
) -> DosMoE:
    device = device or torch.device("cpu")
    experts = [load_frozen_expert(spec, device=device) for spec in specs]
    gating = GatingNetwork(
        input_dim=gating_input_dim,
        hidden_dim=gating_hidden_dim,
        num_experts=len(experts),
    ).to(device)
    return DosMoE(gating=gating, experts=experts).to(device)


def demo_training_loop() -> None:
    """Run a minimal training loop on synthetic data to illustrate usage."""

    batch_size, time_steps, features = 8, 12, 16
    device = torch.device("cpu")

    moe_model = build_moe_model(
        specs=[AUTOENCODER_SPEC, CNN_SPEC, LSTM_SPEC],
        gating_input_dim=time_steps * features,
        gating_hidden_dim=64,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(moe_model.gating.parameters(), lr=1e-3)

    # Synthetic training data
    inputs = torch.randn(batch_size, time_steps, features, device=device)
    targets = torch.randint(0, 2, (batch_size,), device=device).float()

    moe_model.train()
    optimizer.zero_grad()
    logits = moe_model(inputs)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()

    print(f"Training step complete. Loss: {loss.item():.4f}")


if __name__ == "__main__":
    demo_training_loop()
