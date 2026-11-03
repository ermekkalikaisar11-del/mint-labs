"""
mint_engine.py  (v10)  -  Курьер: загрузка РЕАЛЬНЫХ данных NASA и первичный осмотр.

Что изменилось по сравнению с v9:
  * грузим ВСЕ кварталы Kepler (долгая каденция), а не первые 5 из списка;
  * каждый квартал нормализуется отдельно, потом они склеиваются по времени;
  * кэш на диске (MINT_CSV_Data): второй запуск не ждёт NASA;
  * быстрый осмотр больше не "угадывает" период и радиус (раньше это было
    случайное число) - только честная статистика данных.
"""
import os
import re
import json
import numpy as np

# Если при скачивании будет ошибка SSL (например, в школьной сети),
# поставь True. По умолчанию проверка сертификатов ВКЛЮЧЕНА (так безопаснее).
INSECURE_SSL = False
if INSECURE_SSL:
    import ssl
    import urllib3
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    import lightkurve as lk
    LIGHTKURVE_AVAILABLE = True
except ImportError:
    LIGHTKURVE_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "MINT_CSV_Data")


class MintAstroEngine:

    @staticmethod
    def _safe_name(name):
        return re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())

    @staticmethod
    def _cache_path(target):
        return os.path.join(DATA_DIR, MintAstroEngine._safe_name(target) + ".npz")

    @staticmethod
    def _search(target, log):
        """Пробуем по очереди: Kepler (долгая каденция) -> TESS SPOC -> что угодно."""
        attempts = [
            ("Kepler, долгая каденция", dict(mission="Kepler", author="Kepler", cadence="long")),
            ("TESS SPOC, 2 минуты", dict(mission="TESS", author="SPOC", exptime=120)),
            ("любые доступные данные", dict()),
        ]
        for label, kwargs in attempts:
            try:
                res = lk.search_lightcurve(target, **kwargs)
            except Exception as e:
                log(f"  ⚠️ Поиск '{label}' не удался: {e}")
                continue
            if len(res) > 0:
                log(f"📦 Найдено наборов данных ({label}): {len(res)}")
                return res
        return None

    @staticmethod
    def download_real_data(target_name, log, status, use_cache=True):
        """Возвращает (time, flux, meta). При неудаче (None, None, None).
        meta: {'star_radius': float|None, 'teff': float|None, 'n_parts': int, 'source': str}"""
        cache = MintAstroEngine._cache_path(target_name)
        if use_cache and os.path.exists(cache):
            try:
                d = np.load(cache, allow_pickle=False)
                meta = json.loads(str(d["meta"]))
                log(f"💾 Данные загружены из кэша: {os.path.basename(cache)}")
                log(f"✅ Точек фотометрии: {len(d['time']):,}")
                return d["time"], d["flux"], meta
            except Exception as e:
                log(f"  ⚠️ Кэш повреждён ({e}), скачиваем заново.")

        if not LIGHTKURVE_AVAILABLE:
            log("❌ Библиотека lightkurve не установлена (pip install lightkurve).")
            return None, None, None

        status("⏳ Запрос к архиву NASA MAST...")
        log(f"🌐 Поиск данных для: {target_name}...")
        try:
            search = MintAstroEngine._search(target_name, log)
            if search is None:
                log(f"⚠️ Объект '{target_name}' не найден.")
                return None, None, None

            n = len(search)
            log(f"⬇️ Скачиваем все {n} частей (первый раз это занимает несколько минут)...")
            all_t, all_f = [], []
            star_radius = teff = None

            for i in range(n):
                status(f"⏳ Скачивание части {i + 1}/{n}...")
                try:
                    lc = search[i].download(quality_bitmask="default")
                except Exception as e:
                    log(f"  ⚠️ Часть {i + 1}/{n} пропущена: {e}")
                    continue
                if lc is None:
                    continue
                lc = lc.remove_nans().normalize()      # нормализация КАЖДОЙ части отдельно
                t = np.asarray(lc.time.value, dtype=float)
                f = np.asarray(lc.flux.value, dtype=float)
                ok = np.isfinite(t) & np.isfinite(f)
                if ok.sum() < 50:
                    continue
                all_t.append(t[ok])
                all_f.append(f[ok])
                log(f"  📡 Часть {i + 1}/{n}: {ok.sum():,} точек")

                if star_radius is None:
                    meta_lc = getattr(lc, "meta", {}) or {}
                    for key, target_var in (("RADIUS", "r"), ("TEFF", "t")):
                        try:
                            v = float(meta_lc.get(key))
                            if np.isfinite(v) and v > 0:
                                if target_var == "r":
                                    star_radius = v
                                else:
                                    teff = v
                        except (TypeError, ValueError):
                            pass

            if not all_t:
                log("❌ Не удалось скачать ни одной части данных.")
                return None, None, None

            time_arr = np.concatenate(all_t)
            flux_arr = np.concatenate(all_f)
            order = np.argsort(time_arr)
            time_arr, flux_arr = time_arr[order], flux_arr[order]

            meta = {"star_radius": star_radius, "teff": teff,
                    "n_parts": len(all_t), "source": target_name}
            log(f"✅ Загружено! Частей: {len(all_t)}, всего точек: {len(time_arr):,}")

            try:
                os.makedirs(DATA_DIR, exist_ok=True)
                np.savez_compressed(cache, time=time_arr, flux=flux_arr, meta=json.dumps(meta))
                log(f"💾 Сохранено в кэш: {os.path.basename(cache)}")
            except Exception as e:
                log(f"  ⚠️ Кэш не сохранён: {e}")

            return time_arr, flux_arr, meta

        except Exception as e:
            log(f"❌ Критический сбой сети/API: {e}")
            return None, None, None

    @staticmethod
    def quick_look(time, flux):
        """Честная статистика данных. Никаких 'периодов' и 'радиусов' на этом этапе."""
        t = np.asarray(time, dtype=float)
        f = np.asarray(flux, dtype=float)
        order = np.argsort(t)
        t, f = t[order], f[order]
        dt = np.diff(t)
        cadence = float(np.median(dt))
        gaps = int(np.sum(dt > 0.5))
        d = np.diff(f)
        noise = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)   # шум без тренда
        return {
            "n_points": int(len(t)),
            "span_days": float(t[-1] - t[0]),
            "cadence_min": cadence * 24 * 60,
            "n_gaps": gaps,
            "noise_ppm": float(noise * 1e6),
        }
