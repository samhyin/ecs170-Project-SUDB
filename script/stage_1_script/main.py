import runpy
import os
from pathlib import Path


DEFAULT_SCRIPT = "script_mlp.py"


def main():
    script_dir = Path(__file__).resolve().parent
    previous_cwd = Path.cwd()
    try:
        os.chdir(script_dir)
        runpy.run_path(str(script_dir / DEFAULT_SCRIPT), run_name="__main__")
    finally:
        os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
