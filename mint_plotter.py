"""
mint_plotter.py  (v10)  -  Художник: графики внутри окна Tkinter.

Два вида:
  show_curve  - вся кривая блеска; найденные транзиты отмечены цветными треугольниками.
  show_fold   - "яма" планеты: все её транзиты наложены друг на друга (с увеличением),
                жирная линия = медиана по часовым ячейкам. Именно по этому графику
                видно, настоящий ли сигнал: должна быть чёткая ступенька вниз в центре.
"""
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

COLORS = ["#ff9900", "#00bfff", "#ff4d94", "#b388ff", "#ffee58", "#4dd0e1", "#ff7043", "#9ccc65"]


def fold_zoom(time, flux, pl, half_width=3.0):
    """Часы от центра транзита и яркость в окне +-half_width длительностей.
    Для планет со сдвигами времени (TTV) выравниваем по ИЗМЕРЕННЫМ временам транзитов."""
    time = np.asarray(time)
    flux = np.asarray(flux)
    P, t0, d = pl["period"], pl["t0"], pl["duration"]
    tt = np.sort(np.array(pl.get("tt") or [], dtype=float))
    if len(tt) >= 2 and (pl.get("ttv_rms_hours") or 0) > 0.5 or (len(tt) >= 2 and P > 100):
        i = np.clip(np.searchsorted(tt, time), 1, len(tt) - 1)
        left, right = tt[i - 1], tt[i]
        nearest = np.where(np.abs(time - left) < np.abs(time - right), left, right)
        dt = time - nearest
    else:
        dt = ((time - t0 + P / 2) % P) - P / 2
    m = np.abs(dt) < half_width * d
    return dt[m] * 24.0, flux[m]


def bin_median(x, y, width):
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


class MintPlotter:
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.figure = Figure(figsize=(8, 4), facecolor="#0c0c0f")
        self.ax = self.figure.add_subplot(111)
        self._style()
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _style(self):
        ax = self.ax
        ax.set_facecolor("#050507")
        ax.tick_params(colors="white", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#333333")
        ax.grid(True, color="#1a1a2e", linestyle="--", alpha=0.5)

    def clear(self, message=""):
        self.ax.clear()
        self._style()
        if message:
            self.ax.text(0.5, 0.5, message, color="#a0a0a5", ha="center", va="center",
                         transform=self.ax.transAxes)
        self.canvas.draw()

    def show_curve(self, time, flux, title="Кривая блеска", planets=None):
        ax = self.ax
        ax.clear()
        self._style()
        ax.plot(time, flux, ".", color="#00ff66", ms=1.2, alpha=0.7)
        for k, pl in enumerate(planets or []):
            c = COLORS[k % len(COLORS)]
            n0 = int(np.ceil((np.min(time) - pl["t0"]) / pl["period"]))
            n1 = int(np.floor((np.max(time) - pl["t0"]) / pl["period"]))
            times = pl["t0"] + np.arange(n0, n1 + 1) * pl["period"]
            label = f"P={pl['period']:.2f} д" + (f" ({pl['letter']})" if pl.get("letter") else "")
            ax.plot(times, np.full(len(times), 0.97), "v", color=c, ms=7,
                    transform=ax.get_xaxis_transform(), label=label, clip_on=True)
        if planets:
            leg = ax.legend(fontsize=7, facecolor="#0c0c0f", labelcolor="white", loc="lower left")
            leg.get_frame().set_edgecolor("#333333")
        ax.set_xlabel("Время (дни)", color="#a0a0a5", fontsize=9)
        ax.set_ylabel("Яркость", color="#a0a0a5", fontsize=9)
        ax.set_title(title, color="white", fontsize=10)
        self.figure.tight_layout()
        self.canvas.draw()

    def show_fold(self, time, flux, pl, title=None):
        ax = self.ax
        ax.clear()
        self._style()
        x, y = fold_zoom(time, flux, pl)
        ax.plot(x, y, ".", color="#00ff66", ms=2, alpha=0.35)
        bx, by = bin_median(x, y, width=max(0.5, pl["duration"] * 24 / 8))
        if len(bx):
            ax.plot(bx, by, "-o", color="#ff9900", lw=2, ms=3, label="медиана")
            leg = ax.legend(fontsize=8, facecolor="#0c0c0f", labelcolor="white")
            leg.get_frame().set_edgecolor("#333333")
        if len(y):
            lo, hi = np.percentile(y, [0.5, 99.5])
            pad = (hi - lo) * 0.1 + 1e-5
            ax.set_ylim(min(lo, np.min(by) if len(by) else lo) - pad, hi + pad)
        half = pl["duration"] * 12
        ax.axvspan(-half, half, color="#00ff66", alpha=0.07)
        ax.axhline(1.0, color="#555555", lw=0.8, ls="--")
        ax.set_xlabel("Часы от центра транзита (все транзиты наложены)", color="#a0a0a5", fontsize=9)
        ax.set_ylabel("Яркость", color="#a0a0a5", fontsize=9)
        ax.set_title(title or f"Сигнал {pl.get('id', '')}: P = {pl['period']:.4f} д, "
                              f"глубина {pl['depth'] * 100:.3f}%", color="white", fontsize=10)
        self.figure.tight_layout()
        self.canvas.draw()
