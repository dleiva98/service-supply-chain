
import os, sys, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ssc_pipeline.ssc_pipeline import run as ssc_update
ssc_update()
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "nbconvert", "jupyter"], stdout=subprocess.DEVNULL)
nb = os.path.join(os.path.dirname(__file__), "notebooks", "Auto_Report.ipynb")
outdir = os.path.dirname(__file__)
subprocess.check_call([sys.executable, "-m", "nbconvert", "--to", "html", "--execute", "--output-dir", outdir, nb])
print("HTML report:", os.path.join(outdir, "Auto_Report.html"))
