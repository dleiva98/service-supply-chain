# ssc_pipeline/build_report.py
import os, sys, subprocess, textwrap, pathlib, json
import nbformat as nbf

ROOT = os.environ.get("SSC_ROOT", os.getcwd())
NB_SRC = os.path.join("ssc_pipeline", "notebooks", "Auto_Report.ipynb")
NB_TMP = os.path.join("ssc_pipeline", "notebooks", "Auto_Report.CI.ipynb")
OUT_DIR = "ssc_pipeline"  # nbconvert --output-dir

def ensure_main_csv():
    csv = os.path.join(ROOT, "main.csv")
    if not os.path.exists(csv):
        # Si no hay datos aún, genera un HTML mínimo para no romper Pages
        pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(OUT_DIR, "Auto_Report.html"), "w", encoding="utf-8") as f:
            f.write("<html><body><h1>SSC Report</h1><p>No data (main.csv missing).</p></body></html>")
        print("[build_report] main.csv not found, wrote minimal HTML")
        return False
    return True

def write_ci_guarded_notebook():
    """Crea una copia del notebook con una celda 0 que ajusta paths y evita ssc_update en CI."""
    nb = nbf.read(open(NB_SRC, "r", encoding="utf-8"), as_version=4)
    GUARD = textwrap.dedent("""
    # === CI guard (injected by build_report.py) ===
    import os, sys, pathlib
    # Si corre en GitHub Actions, usar workspace; si no, caer a Drive si existe
    if os.environ.get("GITHUB_ACTIONS","") == "true":
        BASE = os.environ.get("SSC_ROOT", os.getcwd())
    else:
        BASE = os.environ.get("SSC_ROOT", "/content/drive/MyDrive/SSC" if os.path.exists("/content/drive/MyDrive/SSC") else ".")
    os.environ["SSC_ROOT"] = BASE
    os.environ["SSC_CACHE_DIR"] = os.path.join(BASE, ".ssc_cache")
    pathlib.Path(os.environ["SSC_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, BASE)

    # En CI NO ejecutes ssc_update(); en Colab sí es útil.
    if os.environ.get("GITHUB_ACTIONS","") != "true":
        try:
            from ssc_pipeline.ssc_pipeline import run as ssc_update
            ssc_update()
        except Exception as e:
            print("[guard] Aviso: ssc_update() no se ejecutó:", e)

    # Carga de datos
    import pandas as pd
    CSV = os.path.join(BASE, "main.csv")
    if not os.path.exists(CSV):
        # deja un DataFrame vacío para que el resto del notebook no falle
        import pandas as pd
        df = pd.DataFrame()
        print("[guard] main.csv no existe en", CSV)
    else:
        df = pd.read_csv(CSV)
        print("[guard] main.csv cargado:", CSV, "| Filas:", len(df))
    """).strip()

    cell = nbf.v4.new_code_cell(GUARD)
    nb.cells.insert(0, cell)
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
    has_csv = ensure_main_csv()
    nb_path = write_ci_guarded_notebook()
    try:
        run_nbconvert(nb_path)
    except subprocess.CalledProcessError as e:
        # Fallback: genera HTML mínimo para que Pages no falle
        pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(OUT_DIR, "Auto_Report.html"), "w", encoding="utf-8") as f:
            f.write("<html><body><h1>Build failed</h1><pre>{}</pre></body></html>".format(str(e)))
        print("[build_report] nbconvert failed, wrote fallback HTML:", e)

if __name__ == "__main__":
    main()
