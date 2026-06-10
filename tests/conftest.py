import sys
from pathlib import Path

# Flat-layout repo: make `import server` work from tests/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
