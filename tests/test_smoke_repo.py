from pathlib import Path

def test_repo_layout_exists():
    must_exist = [
        "scripts/train.py",
        "scripts/inference.py",
        "scripts/evaluate.py",
        "requirements.txt",
        "dvc.yaml",
        "configs",
        "dataset_mini",
        "test_img",
        "outputs",
    ]
    missing = [p for p in must_exist if not Path(p).exists()]
    assert not missing, f"Missing required paths: {missing}"
