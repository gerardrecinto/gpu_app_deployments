import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import train  # noqa: E402


def test_is_distributed_defaults_false(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert train.is_distributed() is False


def test_is_distributed_true_when_world_size_gt_1(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert train.is_distributed() is True


def test_is_distributed_false_when_world_size_is_1(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert train.is_distributed() is False


def test_build_model_shape():
    args = train.parse_args(["--input-dim", "10"])
    model = train.build_model(args, torch.device("cpu"))
    out = model(torch.randn(4, 10))
    assert out.shape == (4, 1)


def test_seed_is_reproducible(tmp_path):
    out_a = tmp_path / "a.pt"
    out_b = tmp_path / "b.pt"

    for out in (out_a, out_b):
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent.parent / "train.py"),
                "--epochs",
                "2",
                "--seed",
                "5",
                "--output-model",
                str(out),
            ],
            check=True,
            capture_output=True,
        )

    state_a = torch.load(out_a, weights_only=True)
    state_b = torch.load(out_b, weights_only=True)
    for key in state_a:
        assert torch.equal(state_a[key], state_b[key])


def test_help_exits_cleanly():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent.parent / "train.py"), "--help"],
        capture_output=True,
    )
    assert result.returncode == 0
