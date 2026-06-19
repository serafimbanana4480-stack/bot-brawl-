import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


class EmulatorConfig(BaseModel):
    type: str = Field(..., description="Emulator type, e.g., 'bluestacks' or 'ldplayer'")
    adb_path: str = Field(..., description="Path to adb executable")
    screen_width: int = Field(..., ge=800, description="Screen width in pixels")
    screen_height: int = Field(..., ge=600, description="Screen height in pixels")

class BotConfig(BaseModel):
    emulator: EmulatorConfig
    model_path: str = Field(..., description="Path to the YOLO model file")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Detection confidence threshold")
    random_seed: int = Field(42, description="Seed for randomization in safety system")

def load_config(config_path: str | Path = "config.json") -> BotConfig:
    cfg_path = Path(config_path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    data = json.loads(cfg_path.read_text())
    try:
        return BotConfig(**data)
    except ValidationError as e:
        raise ValueError(f"Invalid config: {e}") from e

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate config.json")
    parser.add_argument("--path", default="config.json", help="Path to config file")
    args = parser.parse_args()
    try:
        cfg = load_config(args.path)
        print("Config is valid:")
        print(cfg.json(indent=2))
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc
