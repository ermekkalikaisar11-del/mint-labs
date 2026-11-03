"""
mint_main.py  (v10)  -  Пульт управления (окно Tkinter).

Вкладка 1 "ДАННЫЕ": загрузка реальных данных NASA, быстрый осмотр, график.
Вкладка 2 "АУДИТ": поиск планет (BLS + проверка + слепой поиск), таблица найденных
сигналов (клик по строке показывает "яму" этой планеты), отчёт HTML и CSV.

Все долгие операции идут в фоновых потоках. Потоки НЕ трогают виджеты напрямую:
они кладут сообщения в очередь, а окно забирает их в главном потоке.
"""
import queue
import threading
import webbrowser
import tkinter as tk
from tkinter import ttk, messagebox

from mint_engine import MintAstroEngine
from mint_validator_core import MintDataValidator
from mint_plotter import MintPlotter
from mint_reporter import MintReporter

BG = "#060608"
PANEL = "#0c0c0f"


class MintCommandStation:
    def __init__(self, root):
        self.root = root
        self.root.title("MINT Command Station v10 - Production Engine")
        self.root.geometry("1250x900")
        self.root.configure(bg=BG)

        self.q = queue.Queue()
        self.time_data = None
        self.flux_data = None
        self.meta = {}
        self.target = ""
        self.validator = None
        self.results = None

        self._build_style()
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.tab_data = tk.Frame(self.notebook, bg=PANEL)
        self.tab_audit = tk.Frame(self.notebook, bg=PANEL)
        self.notebook.add(self.tab_data, text=" 🌌 ДАННЫЕ ")
        self.notebook.add(self.tab_audit, text=" 🧠 АУДИТ ПЛАНЕТ ")
        self._build_data_tab()
        self._build_audit_tab()

        self._log("📡 MINT v10 запущен. Введи имя звезды и нажми «ЗАГРУЗИТЬ ДАННЫЕ».", "data")
        self.root.after(100, self._poll)

    # ------------------------------------------------------------------
    def _build_style(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", background="#121216", foreground="#a0a0a5",
                    font=("Consolas", 10), padding=[20, 6])
        s.map("TNotebook.Tab", background=[("selected", "#00ff66")], foreground=[("selected", "#000000")])
        s.configure("Mint.Horizontal.TProgressbar", troughcolor="#121216", background="#00ff66", borderwidth=0)
        s.configure("Treeview", background=PANEL, foreground="#ffffff", fieldbackground=PANEL,
                    font=("Consolas", 10), rowheight=24, borderwidth=0)
        s.configure("Treeview.Heading", background="#181822", foreground="#a0a0a5", font=("Consolas", 9, "bold"))
        s.map("Treeview", background=[("selected", "#00ff66")], foreground=[("selected", "#000000")])

    def _button(self, parent, text, color, command, state=tk.NORMAL):
        return tk.Button(parent, text=text, font=("Consolas", 10, "bold"), bg=color, fg="#000000",
                         activebackground=color, relief=tk.FLAT, command=command, state=state, padx=8)

    def _console(self, parent, fg, height=None):
        kw = {"height": height} if height else {}
        return tk.Text(parent, bg="#050507", fg=fg, font=("Courier New", 10), wrap=tk.WORD, bd=0,
                       highlightthickness=1, highlightbackground="#181822", **kw)

    # ------------------------------------------------------------------
    # Вкладка 1
    # ------------------------------------------------------------------
    def _build_data_tab(self):
        top = tk.Frame(self.tab_data, bg=PANEL, pady=10)
        top.pack(fill=tk.X)
        tk.Label(top, text="Имя звезды:", fg="#ffffff", bg=PANEL, font=("Consolas", 11)).pack(side=tk.LEFT, padx=15)
        self.entry_target = tk.Entry(top, font=("Consolas", 11), width=18, bg="#181822", fg="#ffffff",
                                     insertbackground="white", bd=1, relief=tk.SOLID)
        self.entry_target.insert(0, "Kepler-90")
        self.entry_target.pack(side=tk.LEFT, padx=5)
        self.use_cache = tk.BooleanVar(value=True)
        tk.Checkbutton(top, text="использовать кэш", variable=self.use_cache, bg=PANEL, fg="#a0a0a5",
                       selectcolor="#181822", activebackground=PANEL, activeforeground="#ffffff",
                       font=("Consolas", 9)).pack(side=tk.LEFT, padx=10)
        self.btn_download = self._button(top, "ЗАГРУЗИТЬ ДАННЫЕ", "#00ff66", self.start_download)
        self.btn_download.pack(side=tk.LEFT, padx=10)

        self.status_label = tk.Label(self.tab_data, text="Система готова.", fg="#a0a0a5", bg=PANEL,
                                     font=("Consolas", 10), anchor="w")
        self.status_label.pack(fill=tk.X, padx=15, pady=(0, 5))

        gf = tk.Frame(self.tab_data, bg="#050507", bd=1, relief=tk.SOLID)
        gf.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)
        self.plot_data = MintPlotter(gf)
        self.plot_data.clear("Здесь появится кривая блеска")

        self.console_data = self._console(self.tab_data, "#00ff66", height=10)
        self.console_data.pack(fill=tk.X, padx=15, pady=5)
        self.progress_data = ttk.Progressbar(self.tab_data, style="Mint.Horizontal.TProgressbar", mode="indeterminate")
        self.progress_data.pack(fill=tk.X, padx=15, pady=(0, 10))

    # ------------------------------------------------------------------
    # Вкладка 2
    # ------------------------------------------------------------------
    def _build_audit_tab(self):
        top = tk.Frame(self.tab_audit, bg=PANEL, pady=10)
        top.pack(fill=tk.X)
        self.btn_audit = self._button(top, "🧠 ЗАПУСТИТЬ ГЛУБОКИЙ АУДИТ", "#ff9900", self.start_audit, tk.DISABLED)
        self.btn_audit.pack(side=tk.LEFT, padx=15)
        tk.Label(top, text="Радиус звезды, R☉:", fg="#ffffff", bg=PANEL, font=("Consolas", 10)).pack(side=tk.LEFT)
        self.entry_radius = tk.Entry(top, font=("Consolas", 11), width=6, bg="#181822", fg="#ffffff",
                                     insertbackground="white", bd=1, relief=tk.SOLID)
        self.entry_radius.insert(0, "1.0")
        self.entry_radius.pack(side=tk.LEFT, padx=8)
        self.btn_curve = self._button(top, "ВСЯ КРИВАЯ", "#555566", self.show_whole_curve, tk.DISABLED)
        self.btn_curve.pack(side=tk.LEFT, padx=5)
        self.btn_report = self._button(top, "📄 ОТЧЁТ (HTML)", "#00bfff", self.save_report, tk.DISABLED)
        self.btn_report.pack(side=tk.LEFT, padx=5)
        self.btn_csv = self._button(top, "📊 CSV", "#00bfff", self.save_csv, tk.DISABLED)
        self.btn_csv.pack(side=tk.LEFT, padx=5)

        mid = tk.Frame(self.tab_audit, bg=PANEL)
        mid.pack(fill=tk.BOTH, expand=True, padx=15)

        cols = ("id", "period", "depth", "radius", "snr", "transits", "verdict")
        heads = ("№", "Период, д", "Глубина %", "R, R⊕", "SNR", "Транзиты", "Вердикт")
        widths = (60, 95, 85, 65, 65, 80, 210)
        self.tree = ttk.Treeview(mid, columns=cols, show="headings", height=8, selectmode="browse")
        for c, h, w in zip(cols, heads, widths):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill=tk.X, pady=(0, 6))
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        gf = tk.Frame(mid, bg="#050507", bd=1, relief=tk.SOLID)
        gf.pack(fill=tk.BOTH, expand=True)
        self.plot_audit = MintPlotter(gf)
        self.plot_audit.clear("Запусти аудит, потом кликни по строке таблицы")

        self.console_audit = self._console(self.tab_audit, "#ff9900", height=11)
        self.console_audit.pack(fill=tk.X, padx=15, pady=6)
        self.progress_audit = ttk.Progressbar(self.tab_audit, style="Mint.Horizontal.TProgressbar", mode="indeterminate")
        self.progress_audit.pack(fill=tk.X, padx=15, pady=(0, 10))

    # ------------------------------------------------------------------
    # Очередь сообщений от потоков
    # ------------------------------------------------------------------
    def _logger(self, which):
        return lambda text: self.q.put(("log", which, str(text)))

    def _log(self, text, which):
        box = self.console_data if which == "data" else self.console_audit
        box.insert(tk.END, text + "\n")
        box.see(tk.END)

    def _poll(self):
        try:
            while True:
                item = self.q.get_nowait()
                kind = item[0]
                if kind == "log":
                    self._log(item[2], item[1])
                elif kind == "status":
                    self.status_label.config(text=item[1])
                elif kind == "download_done":
                    self._on_download_done(*item[1:])
                elif kind == "download_fail":
                    self._on_download_fail(item[1])
                elif kind == "audit_done":
                    self._on_audit_done(item[1], item[2])
                elif kind == "audit_fail":
                    self._on_audit_fail(item[1])
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    # ------------------------------------------------------------------
    # Загрузка данных
    # ------------------------------------------------------------------
    def start_download(self):
        target = self.entry_target.get().strip()
        if not target:
            messagebox.showwarning("MINT", "Поле имени звезды пустое!")
            return
        self.btn_download.config(state=tk.DISABLED)
        self.btn_audit.config(state=tk.DISABLED)
        self.progress_data.start(15)
        self.console_data.delete(1.0, tk.END)
        self.plot_data.clear("Загрузка...")
        threading.Thread(target=self._download_worker, args=(target, self.use_cache.get()), daemon=True).start()

    def _download_worker(self, target, use_cache):
        log = self._logger("data")
        status = lambda s: self.q.put(("status", s))
        try:
            t, f, meta = MintAstroEngine.download_real_data(target, log, status, use_cache)
            if t is None:
                self.q.put(("download_fail", "Не удалось загрузить данные."))
                return
            stats = MintAstroEngine.quick_look(t, f)
            self.q.put(("download_done", t, f, meta, stats, target))
        except Exception as e:
            self.q.put(("download_fail", f"{type(e).__name__}: {e}"))

    def _on_download_done(self, t, f, meta, stats, target):
        self.progress_data.stop()
        self.btn_download.config(state=tk.NORMAL)
        self.btn_audit.config(state=tk.NORMAL)
        self.time_data, self.flux_data, self.meta, self.target = t, f, meta, target
        self.status_label.config(text="✅ Данные загружены. Переходи на вкладку «АУДИТ ПЛАНЕТ».")
        self._log("\n=== ОСМОТР ДАННЫХ ===", "data")
        self._log(f"Точек: {stats['n_points']:,} | длина: {stats['span_days']:.0f} д | "
                  f"каденция: {stats['cadence_min']:.1f} мин", "data")
        self._log(f"Шум: ~{stats['noise_ppm']:.0f} ppm | разрывов в данных: {stats['n_gaps']}", "data")
        self._log("Период и радиус здесь НЕ считаются: их ищет глубокий аудит.", "data")
        radius = meta.get("star_radius") if meta else None
        self.entry_radius.delete(0, tk.END)
        self.entry_radius.insert(0, f"{radius:.2f}" if radius else "1.0")
        if radius:
            self._log(f"Радиус звезды из заголовка данных: {radius:.2f} R☉ (можно изменить на вкладке аудита)", "data")
        self.plot_data.show_curve(t, f, title=f"Сырые данные: {target}")

    def _on_download_fail(self, msg):
        self.progress_data.stop()
        self.btn_download.config(state=tk.NORMAL)
        self.status_label.config(text="❌ Ошибка загрузки.")
        self._log(f"❌ {msg}", "data")

    # ------------------------------------------------------------------
    # Аудит
    # ------------------------------------------------------------------
    def start_audit(self):
        if self.time_data is None:
            messagebox.showwarning("MINT", "Сначала загрузи данные на первой вкладке!")
            return
        try:
            radius = float(self.entry_radius.get().replace(",", "."))
            if not (0.1 <= radius <= 50):
                raise ValueError
        except ValueError:
            messagebox.showwarning("MINT", "Радиус звезды должен быть числом от 0.1 до 50.")
            return
        for b in (self.btn_audit, self.btn_report, self.btn_csv, self.btn_curve):
            b.config(state=tk.DISABLED)
        self.progress_audit.start(15)
        self.console_audit.delete(1.0, tk.END)
        for i in self.tree.get_children():
            self.tree.delete(i)
        self.plot_audit.clear("Идёт аудит... это может занять несколько минут")
        threading.Thread(target=self._audit_worker, args=(radius,), daemon=True).start()

    def _audit_worker(self, radius):
        try:
            v = MintDataValidator(self.time_data, self.flux_data, star_radius_sun=radius,
                                  target_name=self.target, log=self._logger("audit"))
            results = v.run_full_validation()
            self.q.put(("audit_done", v, results))
        except Exception as e:
            self.q.put(("audit_fail", f"{type(e).__name__}: {e}"))

    def _on_audit_done(self, validator, results):
        self.progress_audit.stop()
        self.validator, self.results = validator, results
        self.btn_audit.config(state=tk.NORMAL)
        self.btn_curve.config(state=tk.NORMAL)
        if results["n_signals"] > 0:
            self.btn_report.config(state=tk.NORMAL)
            self.btn_csv.config(state=tk.NORMAL)

        for i, p in enumerate(results["planets"]):
            name = f"{p['id']}" + (f" ({p['letter']})" if p.get("letter") else "")
            self.tree.insert("", tk.END, iid=str(i), values=(
                name, f"{p['period']:.4f}", f"{p['depth'] * 100:.4f}", f"{p['radius_earth']:.1f}",
                f"{p['snr']:.1f}", f"{p['n_found']}/{p['n_with_data']}", p["verdict"]))

        self.show_whole_curve()
        if results["n_signals"] > 0:
            self.tree.selection_set("0")
        messagebox.showinfo("MINT", f"Аудит завершён.\n{results['verdict']}")

    def _on_audit_fail(self, msg):
        self.progress_audit.stop()
        self.btn_audit.config(state=tk.NORMAL)
        self._log(f"❌ Ошибка аудита: {msg}", "audit")
        self.plot_audit.clear("Ошибка аудита, см. консоль")

    def show_whole_curve(self):
        if self.validator is None or self.validator.time is None:
            return
        self.plot_audit.show_curve(self.validator.time, self.validator.flux,
                                   title="Очищенная кривая блеска, треугольники = найденные транзиты",
                                   planets=self.results["planets"])

    def _on_select(self, _event=None):
        sel = self.tree.selection()
        if not sel or self.results is None:
            return
        pl = self.results["planets"][int(sel[0])]
        self.plot_audit.show_fold(self.validator.time, self.validator.flux, pl)

    # ------------------------------------------------------------------
    # Отчёты
    # ------------------------------------------------------------------
    def save_report(self):
        if not self.results or self.results["n_signals"] == 0:
            messagebox.showwarning("MINT", "Нет найденных сигналов для отчёта.")
            return
        try:
            path = MintReporter.generate_html_report(self.target, self.validator.time,
                                                     self.validator.flux, self.results)
        except Exception as e:
            messagebox.showerror("MINT", f"Не удалось создать отчёт:\n{e}")
            return
        self._log(f"✅ Отчёт сохранён: {path}", "audit")
        webbrowser.open("file://" + path)

    def save_csv(self):
        if not self.results:
            return
        try:
            path = MintReporter.save_csv(self.target, self.results)
        except Exception as e:
            messagebox.showerror("MINT", f"Не удалось сохранить CSV:\n{e}")
            return
        self._log(f"✅ CSV сохранён: {path}", "audit")


if __name__ == "__main__":
    root = tk.Tk()
    app = MintCommandStation(root)
    root.mainloop()
