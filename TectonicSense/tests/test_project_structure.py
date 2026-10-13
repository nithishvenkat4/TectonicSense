from pathlib import Path


def test_required_directories_exist():
    root = Path(__file__).resolve().parents[1]
    required = [
        "data/raw",
        "data/interim",
        "data/processed",
        "notebooks",
        "src",
        "models",
        "results",
        "configs",
        "api",
        "app",
        "docs",
    ]
    for path in required:
        assert (root / path).exists(), f"Missing required path: {path}"
