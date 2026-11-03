"""
mint_validator_core.py  (v10)  -  Судья: полный конвейер поиска планет.

Этапы:
  0. Очистка: выбросы, медленные изменения яркости звезды (детрендинг по сегментам).
  1. BLS (поиск "коробочных" провалов) в реальных днях, до 8 сигналов по очереди:
     нашли сильнейший -> вырезали его транзиты -> ищем следующий.
  2. Проверка КАЖДОГО сигнала независимым методом (mint_matched_filter):
     видны ли транзиты по отдельности, не сдвинуты ли, видна ли планета
     в обеих половинах данных, не прижаты ли транзиты к краям сегментов.
  3. Слепой поиск по остатку (вдруг BLS что-то пропустил).
  4. Сравнение с каталогом известных планет (если звезда есть в KNOWN_SYSTEMS).

Вердикты честные: ✅ только если независимая проверка тоже подтвердила.
"""
import numpy as np
from astropy.timeseries import BoxLeastSquares
from scipy.ndimage import median_filter
from mint_matched_filter import MintMatchedFilter

# Периоды известных планет (дни) - ТОЛЬКО для проверки программы на знакомой звезде.
# Сверь значения с NASA Exoplanet Archive перед тем как использовать в работе.
KNOWN_SYSTEMS = {
    "kepler-90": {
        "b": 7.008151, "c": 8.719375, "i": 14.44912, "d": 59.73667,
        "e": 91.93913, "f": 124.9144, "g": 210.60697, "h": 331.60059,
    },
}


class MintDataValidator:
    MIN_PERIOD = 1.0          # дни
    MAX_PERIOD = 1300.0       # дни (реально ограничено длиной данных)
    MAX_PLANETS = 8           # сколько сигналов искать по очереди
    SNR_OK = 7.0              # порог "похоже на настоящее" (общепринятый)
    SNR_STOP = 6.0            # ниже этого BLS-сигналы больше не ищем
    DETREND_WINDOW = 3.0      # дни, окно медианного фильтра (>= 3 длительностей транзита)
    GAP_DAYS = 0.5            # разрыв больше этого делит данные на сегменты
    BIN_DAYS = 1.0 / 24.0     # BLS работает на данных, усреднённых до 1 часа (быстрее)
    DURATIONS = np.array([0.05, 0.08, 0.12, 0.18, 0.27, 0.40, 0.60, 0.85])  # дни
    CUT_SCALE = 1.5           # вырезаем +-1.5 длительности вокруг найденных транзитов
    GRID_STEP_SCALE = 1.0     # >1 = грубее сетка периодов и БЫСТРЕЕ (например 2.0)
    BLS_OVERSAMPLE = 4
    EDGE_DAYS = 1.0           # транзит ближе этого к краю сегмента - подозрительный
    ENABLE_BLIND = True
    BLIND_SNR = 6.0

    def __init__(self, time, flux, star_radius_sun=1.0, target_name="", log=print):
        self.raw_time = np.asarray(time, dtype=float)
        self.raw_flux = np.asarray(flux, dtype=float)
        self.star_radius_sun = float(star_radius_sun)
        self.target_name = target_name
        self.log = log
        self.time = None          # очищенное время
        self.flux = None          # очищенный и выпрямленный поток
        self.sigma = None
        self.edges = np.array([])
        self.planets = []
        self.results = {}

    # ------------------------------------------------------------------
    # 0. ПОДГОТОВКА ДАННЫХ
    # ------------------------------------------------------------------
    def _segments(self, t):
        breaks = np.where(np.diff(t) > self.GAP_DAYS)[0] + 1
        return np.split(np.arange(len(t)), breaks)

    def _detrend(self, t, f):
        trend = np.ones_like(f)
        for seg in self._segments(t):
            if len(seg) < 5:
                trend[seg] = np.median(f[seg])
                continue
            cadence = np.median(np.diff(t[seg]))
            win = max(int(round(self.DETREND_WINDOW / cadence)), 5)
            if win % 2 == 0:
                win += 1
            if len(seg) <= win:
                trend[seg] = np.median(f[seg])
            else:
                trend[seg] = median_filter(f[seg], size=win, mode="nearest")
        return f / trend

    def _remove_outliers(self, t, f):
        """Убираем выбросы вверх и ОДИНОЧНЫЕ провалы вниз (космические лучи и т.п.).
        Настоящий транзит длится много точек подряд, его не трогаем."""
        resid = f - 1.0
        sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        if sigma == 0:
            return t, f, 0.0
        bad_up = resid > 4 * sigma
        low = resid < -5 * sigma
        prev_ok = np.r_[True, resid[:-1] > -2 * sigma]
        next_ok = np.r_[resid[1:] > -2 * sigma, True]
        keep = ~(bad_up | (low & prev_ok & next_ok))
        removed = int((~keep).sum())
        if removed:
            self.log(f"🧹 Удалено выбросов: {removed}")
        return t[keep], f[keep], sigma

    def prepare(self):
        t, f = self.raw_time, self.raw_flux
        ok = np.isfinite(t) & np.isfinite(f)
        t, f = t[ok], f[ok]
        t, idx = np.unique(t, return_index=True)          # сортировка + удаление дублей
        f = f[idx] / np.median(f[idx])
        if len(t) < 200:
            raise ValueError("Слишком мало точек для анализа")
        f = self._detrend(t, f)
        t, f, sigma = self._remove_outliers(t, f)
        self.time, self.flux, self.sigma = t, f, float(sigma)
        segs = self._segments(t)
        self.edges = np.array([x for s in segs for x in (t[s[0]], t[s[-1]])])
        self.log(f"📊 Точек после очистки: {len(t):,}, шум ~ {sigma * 1e6:.0f} ppm, "
                 f"длина данных {t[-1] - t[0]:.0f} д, сегментов {len(segs)}")

    # ------------------------------------------------------------------
    # 1. BLS
    # ------------------------------------------------------------------
    def _bin(self, t, f):
        idx = np.floor((t - t[0]) / self.BIN_DAYS).astype(np.int64)
        _, inv, cnt = np.unique(idx, return_inverse=True, return_counts=True)
        tb = np.bincount(inv, weights=t) / cnt
        fb = np.bincount(inv, weights=f) / cnt
        return tb, fb, self.sigma / np.sqrt(cnt)

    def _period_grid(self, baseline, pmax):
        """Сетка периодов: шаг подобран так, чтобы за всё время наблюдений транзит
        не "уезжал" больше чем на треть своей длительности. Равномерна по u = P^(-1/3)."""
        k = 0.54 * self.GRID_STEP_SCALE / (3.0 * baseline * 365.0 ** (1 / 3))   # dP = k * P^(4/3)
        u_hi = self.MIN_PERIOD ** (-1 / 3)
        u_lo = pmax ** (-1 / 3)
        u = np.arange(u_lo, u_hi, k / 3.0)
        return (u ** -3.0)[::-1]

    def _run_bls(self, t, f, pmax):
        tb, fb, dyb = self._bin(t, f)
        baseline = tb.max() - tb.min()
        periods = self._period_grid(baseline, pmax)
        durations = self.DURATIONS[self.DURATIONS < periods.min()]
        model = BoxLeastSquares(tb, fb, dy=dyb)

        kw = dict(objective="snr", oversample=self.BLS_OVERSAMPLE)
        if pmax / (durations.min() / self.BLS_OVERSAMPLE) > 50000:
            kw["use_long_mode"] = True
        try:
            res = model.power(periods, durations, **kw)
        except TypeError:
            kw.pop("use_long_mode", None)
            res = model.power(periods, durations, **kw)

        i = int(np.argmax(res.power))
        return {"period": float(res.period[i]), "duration": float(res.duration[i]),
                "t0": float(res.transit_time[i]), "depth": float(res.depth[i]),
                "snr": float(res.depth_snr[i]), "n_periods": int(len(periods))}

    def _cut(self, t, f, pl):
        ph = ((t - pl["t0"] + pl["period"] / 2) % pl["period"]) - pl["period"] / 2
        keep = np.abs(ph) > self.CUT_SCALE * pl["duration"]
        return t[keep], f[keep]

    # ------------------------------------------------------------------
    # 2. ПРОВЕРКА И ВЕРДИКТ
    # ------------------------------------------------------------------
    @staticmethod
    def _spec(pl):
        return {"period": pl["period"], "t0": pl["t0"], "duration": pl["duration"]}

    def _odd_even_sigma(self, rows):
        f = [r for r in rows if r["status"] == "найден" and r["depth"] and r["depth"] > 0]
        odd = [r for r in f if r["n"] % 2]
        even = [r for r in f if r["n"] % 2 == 0]
        if len(f) < 4 or len(odd) < 2 or len(even) < 2:
            return 0.0

        def wmean(rs):
            w = np.array([(r["snr"] / r["depth"]) ** 2 for r in rs])
            d = np.array([r["depth"] for r in rs])
            return (w * d).sum() / w.sum(), 1.0 / np.sqrt(w.sum())

        (do, eo), (de, ee) = wmean(odd), wmean(even)
        return float(abs(do - de) / np.hypot(eo, ee))

    def _edge_fraction(self, rows):
        f = [r for r in rows if r["status"] == "найден"]
        if not f or len(self.edges) == 0:
            return 0.0
        near = [np.min(np.abs(self.edges - r["t_meas"])) < self.EDGE_DAYS for r in f]
        return float(np.mean(near))

    def _make_planet(self, base, ver, source):
        rows = ver["table"]
        found_rows = [r for r in rows if r["status"] == "найден"]
        period, t0 = base["period"], base["t0"]
        if ver["period_fit"] and ver["n_found"] >= 3 and abs(ver["period_fit"] / period - 1) < 0.02:
            period, t0 = ver["period_fit"], ver["t0_fit"]

        depth = base["depth"]
        if len(found_rows) >= 2 and np.median([r["snr"] for r in found_rows]) >= 8:
            depth = float(np.median([r["depth"] for r in found_rows]))   # точнее при сдвигах TTV
        radius = float(np.sqrt(max(depth, 0)) * self.star_radius_sun * 109.1)

        pl = {"period": float(period), "period_bls": float(base["period"]), "t0": float(t0),
              "duration": float(base["duration"]), "depth": float(depth), "radius_earth": radius,
              "snr": float(base["snr"]), "source": source, "letter": "",
              "n_expected": ver["n_expected"], "n_found": ver["n_found"],
              "n_with_data": ver["n_with_data"], "ttv_rms_hours": ver["ttv_rms_hours"],
              "stack_snr": ver["stack_snr"], "half_snr": list(ver["half_snr"]),
              "tt": ver["tt"], "table": rows, "warnings": []}
        pl["verdict"] = self._judge(pl, ver)
        return pl

    def _judge(self, pl, ver):
        w = pl["warnings"]
        found, with_data = ver["n_found"], ver["n_with_data"]
        share = found / with_data if with_data else 0.0
        half = ver["half_snr"]
        halves_ok = all(h is not None and h >= 3.5 for h in half) and with_data >= 4

        if pl["snr"] < self.SNR_OK:
            return "❌ СЛИШКОМ СЛАБЫЙ СИГНАЛ"
        if found < 2 and not halves_ok:
            w.append("Отдельно виден меньше чем 1 транзит из ожидаемых: сигнал не подтверждён.")
            return "❌ НЕ ПОДТВЕРЖДЕНО"

        strong = (found >= 3 and share >= 0.5) or halves_ok
        downgrade = False

        oe = self._odd_even_sigma(pl["table"])
        if oe > 3:
            w.append(f"Чётные и нечётные транзиты разной глубины ({oe:.1f} сигм): возможна двойная звезда.")
            downgrade = True
        if pl["depth"] > 0.02:
            w.append("Провал глубже 2%: слишком глубоко для планеты, вероятно затменная двойная.")
            downgrade = True
        edge = self._edge_fraction(pl["table"])
        if edge >= 0.5 and found >= 2:
            w.append("Большинство транзитов рядом с краями сегментов данных: возможен артефакт прибора.")
            downgrade = True
        if pl["duration"] >= self.DURATIONS.max() * 0.99:
            w.append("Длительность упёрлась в максимум сетки.")
        if found == 2:
            w.append("Виден всего 2 транзита: период - кандидат, нужен третий для подтверждения.")

        if strong and not downgrade:
            return "✅ ПОДТВЕРЖДЕНО"
        if found >= 2 or (ver["stack_snr"] or 0) >= self.SNR_OK:
            return "⚠️ КАНДИДАТ"
        return "❌ НЕ ПОДТВЕРЖДЕНО"

    @staticmethod
    def _is_duplicate(p, planets):
        for q in planets:
            r = p / q["period"]
            for m in (1, 2, 3, 4):
                if abs(r - m) / m < 0.02 or abs(r - 1.0 / m) * m < 0.02:
                    return True
        return False

    def _compare_known(self, planets):
        key = self.target_name.strip().lower().replace(" ", "-")
        known = KNOWN_SYSTEMS.get(key)
        if not known:
            return []
        good = [p for p in planets if p["verdict"][0] in "✅⚠"]
        out = []
        used = set()
        for letter, pk in sorted(known.items(), key=lambda x: x[1]):
            best = None
            for j, p in enumerate(good):
                err = abs(p["period"] - pk) / pk
                if err < 0.02 and j not in used and (best is None or err < best[0]):
                    best = (err, j)
            if best:
                used.add(best[1])
                good[best[1]]["letter"] = letter
                out.append({"letter": letter, "p_known": pk, "p_found": good[best[1]]["period"],
                            "err_pct": best[0] * 100, "found": True,
                            "verdict": good[best[1]]["verdict"]})
            else:
                out.append({"letter": letter, "p_known": pk, "p_found": None,
                            "err_pct": None, "found": False, "verdict": ""})
        return out

    # ------------------------------------------------------------------
    # ГЛАВНЫЙ МЕТОД
    # ------------------------------------------------------------------
    def run_full_validation(self):
        log = self.log
        log("=" * 60)
        log("🧠 ГЛУБОКИЙ АУДИТ MINT v10")
        log("=" * 60)
        self.planets = []
        self.prepare()
        t_all, f_all = self.time, self.flux
        baseline = t_all[-1] - t_all[0]
        pmax = min(self.MAX_PERIOD, baseline * 0.9)

        # --- Этап 1: BLS по очереди ---
        candidates = []
        tw, fw = t_all.copy(), f_all.copy()
        for k in range(self.MAX_PLANETS):
            log(f"\n🔍 Этап 1: BLS, сигнал #{k + 1} (может занять от секунд до нескольких минут)...")
            found = self._run_bls(tw, fw, pmax)
            log(f"   Период {found['period']:.4f} д | глубина {found['depth'] * 100:.4f}% | "
                f"длит. {found['duration'] * 24:.1f} ч | SNR {found['snr']:.1f} "
                f"(сетка: {found['n_periods']:,} периодов)")
            if found["snr"] < self.SNR_STOP:
                log("   Сигнал слабее порога, BLS-поиск закончен.")
                break
            candidates.append(found)
            tw, fw = self._cut(tw, fw, found)
            if len(tw) < 500:
                break

        # --- Этап 2: независимая проверка ---
        log("\n🔬 Этап 2: независимая проверка каждого сигнала (Matched Filter)...")
        mf = MintMatchedFilter(self.time, self.flux, self.sigma, log=log)
        planets = []
        for k, c in enumerate(candidates):
            others = [self._spec(o) for j, o in enumerate(candidates) if j != k]
            ver = mf.masked(others).verify_planet(c["period"], c["t0"], c["duration"])
            pl = self._make_planet(c, ver, "BLS")
            log(f"   => Сигнал P={pl['period']:.4f} д: {pl['verdict']}")
            planets.append(pl)

        # --- Этап 3: слепой поиск по остатку ---
        if self.ENABLE_BLIND and len(self.time) > 1000:
            log("\n🧭 Этап 3: слепой поиск по остатку (твоя идея про промежутки между провалами)...")
            specs = [self._spec(p) for p in planets]
            resid = mf.masked(specs, scale=self.CUT_SCALE)
            dur = 0.25
            events = resid.find_events(dur, snr_min=self.BLIND_SNR)
            groups = resid.blind_periods(events, dur, pmin=3.0, tol_days=0.3, max_groups=4)
            for g in groups:
                if self._is_duplicate(g["period"], planets):
                    continue
                base = {"period": g["period"], "t0": g["events"][0], "duration": dur,
                        "depth": float(np.mean([e["depth"] for e in events if e["t"] in g["events"]]
                                               or [0.0])), "snr": 0.0}
                ver = mf.masked(specs).verify_planet(base["period"], base["t0"], dur)
                base["snr"] = float(ver["stack_snr"] or 0.0)
                pl = self._make_planet(base, ver, "слепой поиск")
                log(f"   => Кандидат P={pl['period']:.4f} д: {pl['verdict']}")
                if pl["verdict"][0] in "✅⚠":
                    planets.append(pl)

        planets.sort(key=lambda p: p["period"])
        for i, p in enumerate(planets, 1):
            p["id"] = i
        self.planets = planets

        known_cmp = self._compare_known(planets)

        # --- Итоги ---
        n_ok = sum(p["verdict"].startswith("✅") for p in planets)
        n_cand = sum(p["verdict"].startswith("⚠") for p in planets)
        if n_ok:
            verdict = f"✅ ПОДТВЕРЖДЕНО СИГНАЛОВ: {n_ok}" + (f", кандидатов: {n_cand}" if n_cand else "")
        elif n_cand:
            verdict = f"⚠️ ТОЛЬКО КАНДИДАТЫ: {n_cand}"
        else:
            verdict = "❌ ПЛАНЕТНЫЕ СИГНАЛЫ НЕ НАЙДЕНЫ"

        log("\n" + "=" * 60)
        log("📋 ИТОГОВАЯ ТАБЛИЦА")
        log(f"{'№':>2} {'Период, д':>11} {'Глубина %':>10} {'R, R⊕':>6} {'SNR':>6} {'транз.':>8}  {'вердикт':<22}")
        for p in planets:
            tag = f" [{p['letter']}]" if p["letter"] else ""
            log(f"{p['id']:>2} {p['period']:>11.4f} {p['depth'] * 100:>10.4f} {p['radius_earth']:>6.1f} "
                f"{p['snr']:>6.1f} {p['n_found']:>3}/{p['n_with_data']:<4}  {p['verdict']}{tag}")
            for w in p["warnings"]:
                log(f"      ⚠ {w}")
        if known_cmp:
            log("\n📚 Сравнение с каталогом (для проверки программы):")
            for k in known_cmp:
                if k["found"]:
                    log(f"   планета {k['letter']}: каталог {k['p_known']:.3f} д, у нас {k['p_found']:.3f} д "
                        f"(ошибка {k['err_pct']:.2f}%)")
                else:
                    log(f"   планета {k['letter']}: каталог {k['p_known']:.3f} д - НЕ найдена")
        log(f"\nВЕРДИКТ: {verdict}")
        log("=" * 60)

        best = max(planets, key=lambda p: p["snr"]) if planets else None
        self.results = {
            "target": self.target_name, "verdict": verdict, "planets": planets,
            "n_signals": len(planets), "known_comparison": known_cmp,
            "final_period": best["period"] if best else 0.0,
            "t0": best["t0"] if best else 0.0,
            "bls_power": best["snr"] if best else 0.0,
            "n_points": int(len(self.time)), "baseline_days": float(baseline),
            "sigma_ppm": float(self.sigma * 1e6), "star_radius_sun": self.star_radius_sun,
        }
        return self.results
