# ssc_pipeline/build_report.py
import os, sys, io, base64, pathlib, datetime as dt
import pandas as pd

# Evitar fallos de backend en CI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("SSC_ROOT", os.getcwd())
OUT_DIR = os.path.join(ROOT, "ssc_pipeline")
CSV = os.path.join(ROOT, "main.csv")

def _now_utc_iso():
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _img_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def _safe_counts(df, col):
    try:
        return df[col].value_counts().reset_index(names=[col, "count"])
    except Exception:
        return pd.DataFrame(columns=[col, "count"])

def _render_table_html(df, max_rows=10):
    if df is None or df.empty:
        return "<p><em>(sin datos)</em></p>"
    return df.head(max_rows).to_html(index=False, border=0)

def build_report():
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(CSV):
        html = f"""<html><body>
        <h1>SSC Report</h1>
        <p><strong>{_now_utc_iso()}</strong> — No se encontró <code>main.csv</code> en el workspace.</p>
        <p>Ejecuta primero el workflow <b>Update SSC Data</b> para generar datos.</p>
        </body></html>"""
        open(os.path.join(OUT_DIR, "Auto_Report.html"), "w", encoding="utf-8").write(html)
        print("[build_report] main.csv no existe; generado HTML mínimo.")
        return

    df = pd.read_csv(CSV)

    # Conteo por fuente
    by_src = _safe_counts(df, "source")

    # Series WITS: USA-WLD por año (si existen)
    line_img = ""
    try:
        wits = df[df["source"].isin(["wits_trade","wits_tariff"])].copy()
        # columna 'time' puede venir como año o "YYYY-MM"
        wits["year_num"] = pd.to_numeric(wits["time"].astype(str).str[:4], errors="coerce")
        q = (
            wits["reporter"].astype(str).str.upper().eq("USA") &
            wits["partner"].astype(str).str.upper().eq("WLD") &
            wits["indicator"].astype(str).isin(["MPRT-TRD-VL","XPRT-TRD-VL"])
        )
        g = wits[q].groupby(["indicator","year_num"], dropna=True)["Value"].sum().reset_index()
        if not g.empty:
            fig, ax = plt.subplots(figsize=(7,4))
            for ind, sub in g.groupby("indicator"):
                sub = sub.dropna(subset=["year_num"]).sort_values("year_num")
                ax.plot(sub["year_num"], sub["Value"], marker="o", label=ind)
            ax.set_title("USA ↔ WLD (WITS) — Totales anuales")
            ax.set_xlabel("Año"); ax.set_ylabel("Valor")
            ax.grid(True, alpha=0.3); ax.legend()
            line_img = _img_to_base64(fig)
    except Exception as e:
        print("[build_report] aviso grafico WITS:", e)

    # Muestras por fuente
    samples = {}
    for s in ["wits_trade","wits_tariff","census_imports","census_exports","bls_ppi"]:
        try:
            sub = df[df["source"]==s].head(10)
            samples[s] = sub
        except Exception:
            samples[s] = pd.DataFrame()

    # Fecha de extracción (si existe) y ahora-UTC
    pulled = ""
    if "pulled_at_utc" in df.columns and df["pulled_at_utc"].notna().any():
        pulled = str(df["pulled_at_utc"].dropna().astype(str).max())

    now_utc = _now_utc_iso()

    # Render HTML
    html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8"/>
<title>SSC Auto Report</title>
<style>
body {{ font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; margin: 24px; }}
h1,h2,h3 {{ margin: 0.4em 0; }}
.section {{ margin-top: 28px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th,td {{ border-bottom: 1px solid #eee; text-align: left; padding: 6px 8px; }}
.badge {{ display:inline-block; background:#f3f4f6; padding:3px 8px; border-radius:8px; margin-right:8px; }}
small {{ color:#666; }}
code {{ background:#f6f8fa; padding:2px 5px; border-radius:4px; }}
</style>
</head>
<body>
<h1>Service Supply Chain — Auto Report</h1>
<p><small>Generado: <b>{now_utc}</b>{' · Último pull: <b>'+pulled+'</b>' if pulled else ''}</small></p>

<div class="section">
  <h2>Resumen por fuente</h2>
  {_render_table_html(by_src)}
</div>

<div class="section">
  <h2>WITS — USA vs Mundo</h2>
  {"<img alt='WITS chart' style='max-width:100%;height:auto' src='"+line_img+"'/>" if line_img else "<p><em>Sin datos suficientes para graficar.</em></p>"}
</div>

<div class="section">
  <h2>Muestras</h2>
  <h3 class="badge">wits_trade</h3>
  {_render_table_html(samples.get('wits_trade'))}
  <h3 class="badge">wits_tariff</h3>
  {_render_table_html(samples.get('wits_tariff'))}
  <h3 class="badge">census_imports</h3>
  {_render_table_html(samples.get('census_imports'))}
  <h3 class="badge">census_exports</h3>
  {_render_table_html(samples.get('census_exports'))}
  <h3 class="badge">bls_ppi</h3>
  {_render_table_html(samples.get('bls_ppi'))}
</div>

</body>
</html>
"""
    out = os.path.join(OUT_DIR, "Auto_Report.html")
    open(out, "w", encoding="utf-8").write(html)
    print("[build_report] Reporte escrito en:", out)

if __name__ == "__main__":
    build_report()

