# ssc_pipeline/build_report.py
import os, sys, subprocess, textwrap, pathlib
import nbformat as nbf

ROOT = os.environ.get("SSC_ROOT", os.getcwd())
NB_SRC = os.path.join("ssc_pipeline", "notebooks", "Auto_Report.ipynb")
NB_TMP = os.path.join("ssc_pipeline", "notebooks", "Auto_Report.CI.ipynb")
OUT_DIR = "ssc_pipeline"  # destino de nbconvert --output-dir

def ensure_main_csv():
    csv = os.path.join(ROOT, "main.csv")
    if not os.path.exists(csv):
        pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(OUT_DIR, "Auto_Report.html"), "w", encoding="utf-8") as f:
            f.write("<html><body><h1>SSC Report</h1><p>No data (main.csv missing).</p></body></html>")
        print("[build_report] main.csv not found, wrote minimal HTML")
        return False
    return True

def write_ci_guarded_notebook():
    """Crea una copia con celda 0 que:
       - fija rutas en CI
       - NO corre ssc_update() en CI
       - simula google.colab.drive.mount() para que no falle
    """
    nb = nbf.read(open(NB_SRC, "r", encoding="utf-8"), as_version=4)
    GUARD = textwrap.dedent(r"""
    # === CI guard (injected by build_report.py) ===
    import os, sys, pathlib, types
    IN_CI = os.environ.get("GITHUB_ACTIONS","") == "true"

    if IN_CI:
        BASE = os.environ.get("SSC_ROOT", os.getcwd())
        # inyecta módulos "falsos" para evitar fallos de Colab
        google_mod = types.ModuleType("google")
        colab_mod = types.ModuleType("google.colab")
        class _Drive:
            @staticmethod
            def mount(*args, **kwargs):
                print("[guard] google.colab.drive.mount() skipped in CI")
        colab_mod.drive = _Drive()
        google_mod.colab = colab_mod
        sys.modules["google"] = google_mod
        sys.modules["google.colab"] = colab_mod
    else:
        BASE = os.environ.get("SSC_ROOT", "/content/drive/MyDrive/SSC" if os.path.exists("/content/drive/MyDrive/SSC") else ".")

    os.environ["SSC_ROOT"] = BASE
    os.environ["SSC_CACHE_DIR"] = os.path.join(BASE, ".ssc_cache")
    pathlib.Path(os.environ["SSC_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, BASE)

    # En CI NO ejecutes ssc_update(); en Colab sí es útil.
    if not IN_CI:
        try:
            from ssc_pipeline.ssc_pipeline import run as ssc_update
            ssc_update()
        except Exception as e:
            print("[guard] Aviso: ssc_update() no se ejecutó:", e)

    # Carga datos de main.csv (si no existe, deja df vacío)
    import pandas as pd
    CSV = os.path.join(BASE, "main.csv")
    if not os.path.exists(CSV):
        df = pd.DataFrame()
        print("[guard] main.csv no existe en", CSV)
    else:
        df = pd.read_csv(CSV)
        print("[guard] main.csv cargado:", CSV, "| Filas:", len(df))
    """).strip()

    nb.cells.insert(0, nbf.v4.new_code_cell(GUARD))
    nbf.write(nb, open(NB_TMP, "w", encoding="utf-8"))
    print("[build_report] CI-guarded notebook written ->", NB_TMP)
    return NB_TMP

def run_nbconvert(nb_path):
    cmd = [
        sys.executable, "-m", "nbconvert",
        "--to", "html",
        "--execute",
        "--output-dir", OUT_DIR,
        nb_path
    ]
    print("[build_report] running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("[build_report] nbconvert OK")

def main():
    ensure_main_csv()   # aunque no exista, publicamos placeholder
    nb_path = write_ci_guarded_notebook()
    try:
        run_nbconvert(nb_path)
    except subprocess.CalledProcessError as e:
        # Fallback: HTML simple para que Pages no falle
        pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(OUT_DIR, "Auto_Report.html"), "w", encoding="utf-8") as f:
            f.write(f"<html><body><h1>Build failed</h1><pre>{e}</pre></body></html>")
        print("[build_report] nbconvert failed, wrote fallback HTML:", e)

if __name__ == "__main__":
    main()

