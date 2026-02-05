from pathlib import Path

def test_repo_layout_exists():
    must_exist = [
        "train.py",
        "inference.py",
        "evaluate.py",
        "requirements.txt",
        "dvc.yaml",
        "configs",
        "test_img",
        "outputs",
    ]
    missing = [p for p in must_exist if not Path(p).exists()]
    assert not missing, f"Missing required paths: {missing}"
