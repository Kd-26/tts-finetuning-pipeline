from pathlib import Path

from tts_finetune.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_config_paths_are_resolved_from_repository_root() -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    assert config.root == ROOT
    assert config.data.manifest == ROOT / "data/demo/metadata.csv"
    assert config.training.output_dir == ROOT / "artifacts/smoke-checkpoint"
