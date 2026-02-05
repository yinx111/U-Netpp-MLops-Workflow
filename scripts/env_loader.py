import os
import warnings
from pathlib import Path

# Silence noisy deprecation warnings (e.g., pkg_resources in torch)
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")


def _parse_env_line(line: str):
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None, None
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip().strip('"').strip("'")
    return key, value


def load_dotenv(path: str = ".env", override: bool = False) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        key, value = _parse_env_line(raw)
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
