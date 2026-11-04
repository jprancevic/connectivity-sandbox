"""Run the notebook headless with its Settings cell overridden (until the notebook becomes a package,
this is how the tests drive it).

    python run_notebook.py NOTEBOOK NAME=value [NAME=value ...]

Each value is a Python literal, e.g.  DEM_PATH="r'data/dem.tif'"  STREAMS_PATH=None
Needs: pip install nbclient ipykernel
"""
import re
import sys

import nbformat
from nbclient import NotebookClient

nb = nbformat.read(sys.argv[1], as_version=4)
cfg = next(c for c in nb.cells if c.cell_type == "code" and "DEM_PATH" in c.source)
for arg in sys.argv[2:]:
    k, v = arg.split("=", 1)
    cfg.source, n = re.subn(rf"^{k}\s*=.*$", f"{k} = {v}", cfg.source, flags=re.M)
    if n != 1:
        raise SystemExit(f"setting {k} not found in the Settings cell")
NotebookClient(nb, kernel_name="python3", timeout=3600).execute()
for c in nb.cells:
    for o in c.get("outputs", []):
        if o.output_type == "stream":
            print(o.text, end="")
