"""
mint_reporter.py  (v10)  -  Секретарь: HTML-отчёт и CSV.

Отчёт содержит: итоговый вердикт, таблицу найденных сигналов, сравнение с каталогом
(если звезда известна), график всей кривой блеска с отметками транзитов и для каждого
сигнала - "яму" (все транзиты наложены) и таблицу найденных транзитов.

Графики строятся через Figure без pyplot, поэтому не конфликтуют с окном Tkinter.
"""
import os
import io
import csv
import html
import base64
from datetime import datetime
import numpy as np
from matplotlib.figure import Figure

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(BASE_DIR, "MINT_Reports")
COLORS = ["#ff9900", "#00bfff", "#ff4d94", "#b388ff", "#ffee58", "#4dd0e1", "#ff7043", "#9ccc65"]


def _fold_zoom(time, flux, pl, half_width=3.0):
    P, t0, d = pl["period"], pl["t0"], pl["duration"]
    tt = np.sort(np.array(pl.get("tt") or [], dtype=float))
    if (len(tt) >= 2 and (pl.get("ttv_rms_hours") or 0) > 0.5) or (len(tt) >= 2 and P > 100):
        i = np.clip(np.searchsorted(tt, time), 1, len(tt) - 1)
        left, right = tt[i - 1], tt[i]
        nearest = np.where(np.abs(time - left) < np.abs(time - right), left, right)
        dt = time - nearest
    else:
        dt = ((time - t0 + P / 2) % P) - P / 2
    m = np.abs(dt) < half_width * d
    return dt[m] * 24.0, flux[m]


def _bin_median(x, y, width):
    if len(x) == 0:
        return np.array([]), np.array([])
    edges = np.arange(x.min(), x.max() + width, width)
    idx = np.digitize(x, edges)
    xs, ys = [], []
    for k in np.unique(idx):
        sel = idx == k
        if sel.sum() >= 3:
            xs.append(x[sel].mean())
            ys.append(np.median(y[sel]))
    return np.array(xs), np.array(ys)


def _style(ax):
    ax.set_facecolor("#050507")
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#333333")
    ax.grid(True, color="#1a1a2e", linestyle="--", alpha=0.5)


def _to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class MintReporter:

    @staticmethod
    def _curve_image(time, flux, planets):
        fig = Figure(figsize=(10, 3.6), facecolor="#0c0c0f")
        ax = fig.add_subplot(111)
        _style(ax)
        ax.plot(time, flux, ".", color="#00ff66", ms=1.2, alpha=0.7)
        for k, pl in enumerate(planets):
            n0 = int(np.ceil((time.min() - pl["t0"]) / pl["period"]))
            n1 = int(np.floor((time.max() - pl["t0"]) / pl["period"]))
            times = pl["t0"] + np.arange(n0, n1 + 1) * pl["period"]
            ax.plot(times, np.full(len(times), 0.97), "v", color=COLORS[k % len(COLORS)], ms=7,
                    transform=ax.get_xaxis_transform(), label=f"P={pl['period']:.2f} д")
        if planets:
            leg = ax.legend(fontsize=7, facecolor="#0c0c0f", labelcolor="white", loc="lower left")
            leg.get_frame().set_edgecolor("#333333")
        ax.set_xlabel("Время (дни)", color="#a0a0a5")
        ax.set_ylabel("Яркость (очищенная)", color="#a0a0a5")
        ax.set_title("Кривая блеска; треугольники = ожидаемые транзиты", color="white", fontsize=10)
        return _to_b64(fig)

    @staticmethod
    def _fold_image(time, flux, pl):
        fig = Figure(figsize=(10, 3.6), facecolor="#0c0c0f")
        ax = fig.add_subplot(111)
        _style(ax)
        x, y = _fold_zoom(time, flux, pl)
        ax.plot(x, y, ".", color="#00ff66", ms=2, alpha=0.35)
        bx, by = _bin_median(x, y, max(0.5, pl["duration"] * 24 / 8))
        if len(bx):
            ax.plot(bx, by, "-o", color="#ff9900", lw=2, ms=3, label="медиана")
            leg = ax.legend(fontsize=8, facecolor="#0c0c0f", labelcolor="white")
            leg.get_frame().set_edgecolor("#333333")
        if len(y):
            lo, hi = np.percentile(y, [0.5, 99.5])
            pad = (hi - lo) * 0.1 + 1e-5
            ax.set_ylim(min(lo, by.min() if len(by) else lo) - pad, hi + pad)
        ax.axvspan(-pl["duration"] * 12, pl["duration"] * 12, color="#00ff66", alpha=0.07)
        ax.axhline(1.0, color="#555555", lw=0.8, ls="--")
        ax.set_xlabel("Часы от центра транзита (все транзиты наложены)", color="#a0a0a5")
        ax.set_ylabel("Яркость", color="#a0a0a5")
        ax.set_title(f"P = {pl['period']:.4f} д", color="white", fontsize=10)
        return _to_b64(fig)

    @staticmethod
    def generate_html_report(star_name, time, flux, results, out_dir=None, filename=None):
        """time, flux - ОЧИЩЕННЫЕ данные (validator.time, validator.flux)."""
        time = np.asarray(time)
        flux = np.asarray(flux)
        planets = results.get("planets", [])
        esc = html.escape

        verdict = results.get("verdict", "НЕИЗВЕСТНО")
        color = "#00ff66" if verdict.startswith("✅") else "#ff9900" if verdict.startswith("⚠") else "#ff2222"

        def vcolor(v):
            return "#00ff66" if v.startswith("✅") else "#ff9900" if v.startswith("⚠") else "#ff5555"

        rows = ""
        for p in planets:
            letter = f" ({esc(p['letter'])})" if p.get("letter") else ""
            ttv = "-" if p.get("ttv_rms_hours") is None else f"{p['ttv_rms_hours']:.2f} ч"
            rows += (f"<tr><td>{p['id']}{letter}</td><td>{p['period']:.4f}</td><td>{p['depth'] * 100:.4f}</td>"
                     f"<td>{p['radius_earth']:.1f}</td><td>{p['snr']:.1f}</td>"
                     f"<td>{p['n_found']} / {p['n_with_data']}</td><td>{ttv}</td>"
                     f"<td style='color:{vcolor(p['verdict'])}'>{esc(p['verdict'])}</td></tr>")

        known_html = ""
        if results.get("known_comparison"):
            kr = ""
            for k in results["known_comparison"]:
                if k["found"]:
                    kr += (f"<tr><td>{esc(k['letter'])}</td><td>{k['p_known']:.3f}</td>"
                           f"<td>{k['p_found']:.3f}</td><td>{k['err_pct']:.2f}%</td>"
                           f"<td style='color:#00ff66'>найдена</td></tr>")
                else:
                    kr += (f"<tr><td>{esc(k['letter'])}</td><td>{k['p_known']:.3f}</td><td>-</td><td>-</td>"
                           f"<td style='color:#ff5555'>не найдена</td></tr>")
            known_html = ("<div class='card'><h2>Сравнение с каталогом известных планет</h2>"
                          "<table><tr><th>Планета</th><th>Период (каталог), д</th><th>Период (MINT), д</th>"
                          f"<th>Ошибка</th><th>Статус</th></tr>{kr}</table></div>")

        cards = ""
        for p in planets:
            img = MintReporter._fold_image(time, flux, p)
            warns = "".join(f"<li>{esc(w)}</li>" for w in p.get("warnings", []))
            found = [r for r in p.get("table", []) if r["status"] == "найден"][:25]
            trs = "".join(
                f"<tr><td>{r['n']}</td><td>{r['t_pred']:.2f}</td><td>{r['t_meas']:.2f}</td>"
                f"<td>{(r['t_meas'] - r['t_pred']) * 24:+.1f}</td><td>{r['depth'] * 100:.4f}</td>"
                f"<td>{r['snr']:.1f}</td></tr>" for r in found)
            tbl = ("<table><tr><th>№</th><th>Ожидали</th><th>Нашли</th><th>Сдвиг, ч</th>"
                   f"<th>Глубина, %</th><th>SNR</th></tr>{trs}</table>") if found else ""
            cards += (f"<div class='card'><h2>Сигнал {p['id']}: P = {p['period']:.4f} д "
                      f"<span style='color:{vcolor(p['verdict'])}'>{esc(p['verdict'])}</span></h2>"
                      f"<p class='muted'>Источник: {esc(p['source'])}. Длительность {p['duration'] * 24:.1f} ч, "
                      f"глубина {p['depth'] * 100:.4f}%, радиус ~{p['radius_earth']:.1f} R⊕ "
                      f"(при радиусе звезды {results.get('star_radius_sun', 1.0):.2f} R☉).</p>"
                      f"{('<ul>' + warns + '</ul>') if warns else ''}"
                      f"<img class='chart-img' src='data:image/png;base64,{img}' alt='fold'>{tbl}</div>")

        curve_img = MintReporter._curve_image(time, flux, planets)

        doc = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>MINT Labs: {esc(star_name)}</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; background:#060608; color:#d0d0e0; margin:0; padding:30px; }}
.container {{ max-width:1000px; margin:0 auto; }}
.header {{ border-bottom:2px solid #00ff66; padding-bottom:15px; margin-bottom:30px; display:flex; justify-content:space-between; align-items:center; }}
.brand {{ font-size:24px; font-weight:bold; color:#00ff66; }}
.card {{ background:#0c0c0f; border:1px solid #1a1a2e; border-radius:8px; padding:25px; margin-bottom:25px; }}
h2 {{ color:#00bfff; margin-top:0; font-size:18px; }}
.verdict-box {{ background:#111; border-left:5px solid {color}; padding:15px; border-radius:4px; margin-bottom:15px; }}
.verdict-text {{ font-size:20px; font-weight:bold; color:{color}; }}
table {{ width:100%; border-collapse:collapse; margin-top:10px; font-size:14px; }}
th,td {{ text-align:left; padding:8px; border-bottom:1px dashed #1a1a2e; }}
th {{ color:#a0a0a5; font-weight:normal; }}
.muted {{ color:#a0a0a5; font-size:13px; }}
.chart-img {{ width:100%; height:auto; border-radius:6px; border:1px solid #1a1a2e; }}
.footer {{ text-align:center; margin-top:40px; color:#5a5a7a; font-size:12px; }}
</style></head><body><div class="container">
<div class="header"><div class="brand">MINT Labs</div><div>Отчёт: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div></div>
<div class="card"><h2>Целевая звезда: {esc(star_name)}</h2>
<div class="verdict-box"><div class="verdict-text">{esc(verdict)}</div></div>
<p class="muted">Точек: {results.get('n_points', 0):,} | длина данных: {results.get('baseline_days', 0):.0f} д |
шум: {results.get('sigma_ppm', 0):.0f} ppm | сигналов: {len(planets)}</p>
<table><tr><th>№</th><th>Период, д</th><th>Глубина, %</th><th>Радиус, R⊕</th><th>SNR</th>
<th>Транзитов (найдено/с данными)</th><th>Разброс времён</th><th>Вердикт</th></tr>{rows}</table></div>
{known_html}
<div class="card"><h2>Вся кривая блеска</h2><img class="chart-img" src="data:image/png;base64,{curve_img}" alt="curve"></div>
{cards}
<div class="footer">MINT Labs. Автоматическая оценка: ✅ значит, что два независимых метода согласны,
но это не научное открытие. Любой кандидат требует независимой проверки.</div>
</div></body></html>"""

        out_dir = out_dir or REPORT_DIR
        os.makedirs(out_dir, exist_ok=True)
        if not filename:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"MINT_Report_{star_name.replace(' ', '_')}_{stamp}.html"
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc)
        return os.path.abspath(path)

    @staticmethod
    def save_csv(star_name, results, out_dir=None):
        out_dir = out_dir or REPORT_DIR
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"MINT_Planets_{star_name.replace(' ', '_')}_{stamp}.csv")
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["id", "letter", "period_d", "t0", "duration_h", "depth_percent",
                        "radius_earth", "snr", "transits_found", "transits_with_data",
                        "ttv_rms_h", "source", "verdict"])
            for p in results.get("planets", []):
                w.writerow([p["id"], p.get("letter", ""), f"{p['period']:.5f}", f"{p['t0']:.4f}",
                            f"{p['duration'] * 24:.2f}", f"{p['depth'] * 100:.5f}",
                            f"{p['radius_earth']:.2f}", f"{p['snr']:.2f}", p["n_found"],
                            p["n_with_data"],
                            "" if p.get("ttv_rms_hours") is None else f"{p['ttv_rms_hours']:.3f}",
                            p["source"], p["verdict"]])
        return os.path.abspath(path)
