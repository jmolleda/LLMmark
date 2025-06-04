import yaml
from pathlib import Path

class Settings:
    def __init__(self, config_path: str = "config.yaml"):
        self._load_config(config_path)

    def _load_config(self, path: str):
        config_file = Path(path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        # Set as attributes
        for key, value in config.items():
            setattr(self, key, value)

    def __repr__(self):
        return f"<Settings {self.__dict__}>"