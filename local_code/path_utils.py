from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
RESULT_ROOT = PROJECT_ROOT / "result"


def resolve_stage_data_dir(stage_name, required_entries):
    stage_root = DATA_ROOT / stage_name
    if not stage_root.exists():
        raise FileNotFoundError(f"Missing stage data directory: {stage_root}")

    candidates = [stage_root, stage_root / stage_name]
    candidates.extend(path for path in stage_root.rglob("*") if path.is_dir())

    checked = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.add(candidate)
        if all((candidate / entry).exists() for entry in required_entries):
            return candidate

    needed = ", ".join(required_entries)
    raise FileNotFoundError(
        f"Could not find a folder inside {stage_root} containing: {needed}"
    )


def ensure_result_dir(stage_name):
    result_dir = RESULT_ROOT / stage_name
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir
