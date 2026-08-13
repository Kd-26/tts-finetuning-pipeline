from pathlib import Path

from tts_finetune.config import load_config
from tts_finetune.validate import validate

ROOT = Path(__file__).resolve().parents[1]


def test_demo_dataset_is_valid() -> None:
    assert validate(load_config(ROOT / "configs/smoke.yaml")) == []
