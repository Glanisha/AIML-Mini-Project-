import sys
from pathlib import Path

# Make the backend's inference package importable as `app.inference` so evaluation
# runs exactly the code path that the API serves
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
