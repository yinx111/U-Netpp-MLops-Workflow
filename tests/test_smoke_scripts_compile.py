import py_compile
from pathlib import Path

def test_scripts_compile():
    scripts = ["train.py", "inference.py", "evaluate.py"]
    for s in scripts:
        p = Path(s)
        if not p.exists():
            continue
        py_compile.compile(str(p), doraise=True)
