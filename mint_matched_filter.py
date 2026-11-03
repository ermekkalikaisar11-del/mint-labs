"""
mint_matched_filter.py  (v10)  -  Детектив: независимая проверка по шаблону.

Идея: окно длиной в один транзит двигаем по всей кривой блеска и считаем,
насколько яркость в окне ниже нормы (в "сигмах шума", SNR).

  verify_planet(period, t0, duration)
      Смотрим КАЖДЫЙ ожидаемый транзит отдельно: когда он реально случился,
      какой глубины, не сдвинут ли по времени (TTV). Плюс проверка "двух половин":
      видна ли планета и в первой, и во второй половине данных.
  find_events + blind_periods
      Слепой поиск: находим провалы без знания периода, потом подбираем периоды
      по промежуткам между ними.
  masked(planets)
      Копия фильтра без транзитов уже найденных планет.

Данные должны быть очищены и выпрямлены (flux ~ 1.0).
"""
import numpy as np
from scipy.signal import find_peaks


class MintMatchedFilter:
    SNR_EVENT = 6.0        # порог для слепого поиска провалов
    SNR_MEASURE = 4.0      # порог, чтобы считать отдельный транзит "найденным"
    MIN_COVERAGE = 0.6     # какую долю окна должны покрывать реальные точки

    def __init__(self, time, flux, sigma=None, log=print):
        t = np.asarray(time, dtype=float)
        f = np.asarray(flux, dtype=float)
        order = np.argsort(t)
        self.t, self.f = t[order], f[order]
        if sigma is None:
            r = self.f - np.median(self.f)
            sigma = 1.4826 * np.median(np.abs(r - np.median(r)))
        self.sigma = float(sigma)
        self.log = log
        self.cadence = float(np.median(np.diff(self.t)))
        # накопленная сумма (flux-1): среднее в любом окне считается мгновенно
        self._cs = np.concatenate([[0.0], np.cumsum(self.f - 1.0)])

    # ------------------------------------------------------------------
    def _window(self, centers, duration):
        """Для каждого центра: (глубина, SNR, покрытие)."""
        centers = np.asarray(centers, dtype=float)
        lo = np.searchsorted(self.t, centers - duration / 2, side="left")
        hi = np.searchsorted(self.t, centers + duration / 2, side="right")
        n = hi - lo
        expected = max(duration / self.cadence, 1.0)
        coverage = n / expected
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = -(self._cs[hi] - self._cs[lo]) / n
            snr = depth / (self.sigma / np.sqrt(n))
        bad = (n == 0) | (coverage < self.MIN_COVERAGE)
        depth = np.where(bad, np.nan, depth)
        snr = np.where(bad, -np.inf, snr)
        return depth, snr, coverage

    def _is_spike(self, tc, duration, depth):
        """Настоящий транзит заполняет всё окно, выброс - 1-2 точки. Сравниваем среднее и медиану."""
        m = (self.t >= tc - duration / 2) & (self.t <= tc + duration / 2)
        if m.sum() < 3:
            return True
        return (1.0 - np.median(self.f[m])) < 0.5 * depth

    def scan(self, duration):
        depth, snr, _ = self._window(self.t, duration)
        return self.t, depth, snr

    def masked(self, planets, scale=1.0):
        """Копия без транзитов других планет (planets: dict с period, t0, duration)."""
        keep = np.ones(len(self.t), dtype=bool)
        for pl in planets:
            ph = ((self.t - pl["t0"] + pl["period"] / 2) % pl["period"]) - pl["period"] / 2
            keep &= np.abs(ph) > scale * pl["duration"]
        return MintMatchedFilter(self.t[keep], self.f[keep], self.sigma, log=self.log)

    # ------------------------------------------------------------------
    # Слепой режим
    # ------------------------------------------------------------------
    def find_events(self, duration, snr_min=None):
        snr_min = self.SNR_EVENT if snr_min is None else snr_min
        tc, depth, snr = self.scan(duration)
        snr_clean = np.where(np.isfinite(snr), snr, -1e9)
        dist = max(1, int(duration / self.cadence))
        peaks, _ = find_peaks(snr_clean, height=snr_min, distance=dist)
        events = []
        for i in peaks:
            if self._is_spike(tc[i], duration, depth[i]):
                continue
            events.append({"t": float(tc[i]), "depth": float(depth[i]), "snr": float(snr[i])})
        self.log(f"🔎 [Фильтр] Окно {duration * 24:.1f} ч: найдено провалов: {len(events)}")
        return events

    def _has_data(self, tc, duration):
        _, snr, _ = self._window([tc], duration)
        return bool(np.isfinite(snr[0]))

    def _score_period(self, p, anchor, times_all, remaining, duration, tol_days):
        tmin, tmax = self.t.min(), self.t.max()
        matched = missed = 0
        for n in range(int(np.ceil((tmin - anchor) / p)), int(np.floor((tmax - anchor) / p)) + 1):
            tp = anchor + n * p
            if not self._has_data(tp, duration):
                continue                                  # в этом месте данных нет
            if np.min(np.abs(times_all - tp)) < tol_days:
                matched += 1
            else:
                missed += 1                               # данные есть, а провала нет
        hit = np.abs(((remaining - anchor + p / 2) % p) - p / 2) < tol_days
        return matched, missed, hit

    def blind_periods(self, events, duration, pmin=1.0, pmax=None, tol_days=0.3, max_groups=5):
        """Подбор периодов по времени провалов. Находит период, объясняющий больше всего
        провалов, убирает их и ищет следующий - так разделяются разные планеты."""
        times_all = np.array(sorted(e["t"] for e in events))
        remaining = times_all.copy()
        pmax = pmax or (self.t.max() - self.t.min())
        groups = []

        for _ in range(max_groups):
            if len(remaining) < 2:
                break
            best = None
            for i in range(len(remaining)):
                for j in range(i + 1, len(remaining)):
                    for k in range(1, 6):
                        p = (remaining[j] - remaining[i]) / k
                        if not (pmin <= p <= pmax):
                            continue
                        anchor = remaining[i]
                        matched, missed, hit = self._score_period(
                            p, anchor, times_all, remaining, duration, tol_days)
                        explained = int(hit.sum())
                        if explained < 2 or missed > 0:
                            continue
                        key = (explained, p)
                        if best is None or key > best[0]:
                            best = (key, p, anchor, hit, matched)
            if best is None:
                break
            _, p, anchor, hit, matched = best
            groups.append({"period": float(p), "events": [float(x) for x in remaining[hit]],
                           "matched": int(matched)})
            remaining = remaining[~hit]

        self.log(f"🧩 [Фильтр] Групп провалов: {len(groups)}, не объяснено провалов: {len(remaining)}")
        for g in groups:
            self.log(f"   P ~ {g['period']:.3f} д, провалов: {len(g['events'])} "
                     f"({', '.join(f'{x:.1f}' for x in g['events'])})")
        return groups

    # ------------------------------------------------------------------
    # Проверка найденного периода по каждому транзиту
    # ------------------------------------------------------------------
    def verify_planet(self, period, t0, duration, search_days=None, verbose=True):
        if search_days is None:
            search_days = max(2 * duration, 0.25)
            if period > 100:
                search_days = max(search_days, 1.0)      # у больших планет заметны сдвиги (TTV)

        tmin, tmax = self.t.min(), self.t.max()
        n_first = int(np.ceil((tmin - t0) / period))
        n_last = int(np.floor((tmax - t0) / period))

        table = []
        for n in range(n_first, n_last + 1):
            tp = t0 + n * period
            m = (self.t >= tp - search_days) & (self.t <= tp + search_days)
            row = {"n": n, "t_pred": float(tp), "t_meas": None, "depth": None,
                   "snr": None, "status": "нет данных"}
            if m.sum() > 0:
                depth, snr, _ = self._window(self.t[m], duration)
                if np.any(np.isfinite(snr)):
                    k = int(np.argmax(np.where(np.isfinite(snr), snr, -np.inf)))
                    tc = float(self.t[m][k])
                    row.update(t_meas=tc, depth=float(depth[k]), snr=float(snr[k]))
                    if self._is_spike(tc, duration, depth[k]):
                        row["status"] = "выброс"
                    elif snr[k] >= self.SNR_MEASURE:
                        row["status"] = "найден"
                    else:
                        row["status"] = "слабый"
            table.append(row)

        found = [r for r in table if r["status"] == "найден"]
        with_data = [r for r in table if r["status"] != "нет данных"]

        stack_depth, stack_snr = self._stack(period, t0, duration)
        half_snr = (self._stack(period, t0, duration, half=0)[1],
                    self._stack(period, t0, duration, half=1)[1])

        result = {"table": table, "n_expected": len(table), "n_found": len(found),
                  "n_with_data": len(with_data), "n_nodata": len(table) - len(with_data),
                  "stack_depth": stack_depth, "stack_snr": stack_snr, "half_snr": half_snr,
                  "period_fit": None, "t0_fit": None, "ttv_rms_hours": None,
                  "tt": [r["t_meas"] for r in found], "duration": duration}

        if len(found) >= 3:
            nn = np.array([r["n"] for r in found], dtype=float)
            tt = np.array([r["t_meas"] for r in found])
            slope, icpt = np.polyfit(nn, tt, 1)
            resid = tt - (icpt + slope * nn)
            result.update(period_fit=float(slope), t0_fit=float(icpt),
                          ttv_rms_hours=float(np.std(resid) * 24))

        if verbose:
            self._print(result, period)
        return result

    def _stack(self, period, t0, duration, half=None):
        """Складываем ожидаемые транзиты (без сдвигов): общая глубина и SNR.
        half=0/1: только первая/вторая половина транзитов (проверка независимости)."""
        tmin, tmax = self.t.min(), self.t.max()
        ns = np.arange(int(np.ceil((tmin - t0) / period)),
                       int(np.floor((tmax - t0) / period)) + 1)
        centers = t0 + ns * period
        lo = np.searchsorted(self.t, centers - duration / 2, side="left")
        hi = np.searchsorted(self.t, centers + duration / 2, side="right")
        n = hi - lo
        ok = n >= self.MIN_COVERAGE * max(duration / self.cadence, 1.0)
        idx = np.where(ok)[0]
        if half is not None:
            if len(idx) < 2:
                return None, None
            cut = len(idx) // 2
            idx = idx[:cut] if half == 0 else idx[cut:]
        total_n = int(n[idx].sum())
        if total_n == 0:
            return None, None
        s = (self._cs[hi[idx]] - self._cs[lo[idx]]).sum()
        depth = -s / total_n
        return float(depth), float(depth / (self.sigma / np.sqrt(total_n)))

    def _print(self, r, period):
        self.log(f"🔬 [Фильтр] Проверка периода {period:.4f} д: ожидалось транзитов "
                 f"{r['n_expected']}, без данных {r['n_nodata']}, найдено по отдельности {r['n_found']}")
        found_rows = [x for x in r["table"] if x["status"] == "найден"]
        for row in found_rows[:12]:
            shift = (row["t_meas"] - row["t_pred"]) * 24
            self.log(f"     #{row['n']:>3}  ждали {row['t_pred']:8.2f} | нашли {row['t_meas']:8.2f} "
                     f"({shift:+5.1f} ч) | глубина {row['depth'] * 100:.4f}% | SNR {row['snr']:.1f}")
        if len(found_rows) > 12:
            self.log(f"     ... и ещё {len(found_rows) - 12}")
        if r["period_fit"]:
            self.log(f"     Уточнённый период: {r['period_fit']:.4f} д, разброс времён {r['ttv_rms_hours']:.2f} ч")
        if r["stack_snr"] is not None:
            h = r["half_snr"]
            hs = " / ".join("-" if x is None else f"{x:.1f}" for x in h)
            self.log(f"     Сложение: глубина {r['stack_depth'] * 100:.4f}%, SNR {r['stack_snr']:.1f} "
                     f"(половины данных: {hs})")
