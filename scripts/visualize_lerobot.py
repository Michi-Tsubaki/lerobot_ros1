#!/usr/bin/env python3
import rospkg
import os
import sys
import subprocess
from pathlib import Path

rospack = rospkg.RosPack()
pkg_path = Path(rospack.get_path("imitation_utils"))

workspace_dir = pkg_path.parents[1]

private_pkg_dir = workspace_dir / "devel" / ".private" / "imitation_utils"
venv_path = private_pkg_dir / "share" / "imitation_utils" / "venv"

venv_bin = venv_path / "bin"
venv_site_packages = venv_path / "lib" / "python3.10" / "site-packages"

os.environ["PATH"] = f"{venv_bin}:{os.environ['PATH']}"

if str(venv_site_packages) not in sys.path:
    sys.path.insert(0, str(venv_site_packages))

python_bin = venv_bin / "python"

cmd = [
    str(python_bin),
    "-m",
    "lerobot.scripts.visualize_dataset",
    "--repo-id",
    "Michi-Tsubaki/nextage_vessel_test1",
    "--episode-index",
    "0",
    "--num-workers",
    "0",
]

print("Running:", " ".join(cmd))
subprocess.call(cmd)
