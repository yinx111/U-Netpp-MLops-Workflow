from pathlib import Path
import yaml

def _load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def test_configs_yaml_parse():
    cfg_dir = Path("configs")
    if not cfg_dir.exists():
        return

    yaml_files = list(cfg_dir.rglob("*.yml")) + list(cfg_dir.rglob("*.yaml"))
    # 如果 configs 目录暂时为空，也别让 CI 失败
    if not yaml_files:
        return

    for p in yaml_files:
        _load_yaml(p)

def test_dvc_yaml_parse():
    p = Path("dvc.yaml")
    if not p.exists():
        return
    _load_yaml(p)
