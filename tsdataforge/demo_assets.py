from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .datasets.series_dataset import SeriesDataset


DATA_DIR = Path(__file__).resolve().parent / "demo_data"


@dataclass(frozen=True)
class DemoScenario:
    scenario_id: str
    title: str
    title_zh: str
    summary: str
    summary_zh: str
    one_liner: str
    one_liner_zh: str
    command_hint: str
    channels: tuple[str, ...]
    tags: tuple[str, ...] = field(default_factory=tuple)
    is_real_public_data: bool = False
    source_label: str = ""
    source_label_zh: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CATALOG: tuple[DemoScenario, ...] = (
    DemoScenario(
        scenario_id="ecg_public",
        title="Public ECG arrhythmia handoff",
        title_zh="公开 ECG 心律失常交接 bundle",
        summary="Real ECG windows derived from the MIT-BIH Arrhythmia Database excerpt bundled by SciPy. Use it to show report-first routing on a genuine biomedical signal.",
        summary_zh="来自 SciPy 所打包 MIT-BIH Arrhythmia Database 片段的真实 ECG 窗口。适合展示真实生物医学信号上的 report-first 路由。",
        one_liner="Best real biomedical demo: one public signal, many windows, fast event/anomaly routing.",
        one_liner_zh="最适合做真实生物医学 demo：一条公开信号切成多窗口，快速走 event/anomaly 路由。",
        command_hint="python -m tsdataforge demo --scenario ecg_public --output ecg_bundle",
        channels=("ecg_mv",),
        tags=("real", "public-data", "medicine", "biosignal", "event-like"),
        is_real_public_data=True,
        source_label="SciPy electrocardiogram (MIT-BIH record 208 excerpt)",
        source_label_zh="SciPy electrocardiogram（MIT-BIH 208 号记录片段）",
    ),
    DemoScenario(
        scenario_id="macro_public",
        title="Public US macro handoff",
        title_zh="公开美国宏观时间序列交接 bundle",
        summary="Real macro windows built from the statsmodels macrodata sample: inflation, unemployment, and Treasury bill rate. Good for regime-style routing without pretending to forecast first.",
        summary_zh="基于 statsmodels macrodata 的真实宏观窗口：通胀、失业和国库券利率。适合先做 regime 风格路由，而不是先假装预测。",
        one_liner="Best real macro demo: explain the regime story before modeling.",
        one_liner_zh="最适合做真实宏观 demo：先把 regime 讲清楚，再决定建模。",
        command_hint="python -m tsdataforge demo --scenario macro_public --output macro_public_bundle",
        channels=("inflation_pct", "unemployment_pct", "tbill_rate_pct"),
        tags=("real", "public-data", "economics", "multivariate", "regime"),
        is_real_public_data=True,
        source_label="statsmodels macrodata sample (US quarterly macro indicators)",
        source_label_zh="statsmodels macrodata 示例（美国季度宏观指标）",
    ),
    DemoScenario(
        scenario_id="climate_public",
        title="Public climate CO₂ handoff",
        title_zh="公开气候 CO₂ 时间序列交接 bundle",
        summary="Real weekly atmospheric CO₂ observations from the classic Mauna Loa series bundled by statsmodels. Good for missingness-aware and trend/seasonality-first routing.",
        summary_zh="来自 statsmodels 打包的经典 Mauna Loa 周度大气 CO₂ 观测。适合 missingness-aware 以及 trend/seasonality-first 路由。",
        one_liner="Best real climate demo: public data, clear trend, seasonality, and missingness in one asset.",
        one_liner_zh="最适合做真实气候 demo：一个资产里同时具有公开数据、清晰趋势、周期和缺失。",
        command_hint="python -m tsdataforge demo --scenario climate_public --output climate_bundle",
        channels=("co2_ppm",),
        tags=("real", "public-data", "climate", "seasonal", "missing"),
        is_real_public_data=True,
        source_label="statsmodels CO₂ sample (weekly atmospheric CO₂)",
        source_label_zh="statsmodels CO₂ 示例（周度大气 CO₂）",
    ),
    DemoScenario(
        scenario_id="sunspots_public",
        title="Public sunspot cycle handoff",
        title_zh="公开太阳黑子周期交接 bundle",
        summary="Real annual sunspot counts from the classic statsmodels sample. Useful as a physical-science example for long cycles and change-aware routing.",
        summary_zh="来自 statsmodels 经典示例的真实年度太阳黑子计数。适合作为物理科学中的长周期与变化感知路由示例。",
        one_liner="Best physical-science demo: real cyclical data with a strong visual story.",
        one_liner_zh="最适合做物理科学 demo：真实周期数据，视觉故事感很强。",
        command_hint="python -m tsdataforge demo --scenario sunspots_public --output sunspots_bundle",
        channels=("sunspots",),
        tags=("real", "public-data", "physical-science", "cyclical"),
        is_real_public_data=True,
        source_label="statsmodels sunspots sample",
        source_label_zh="statsmodels 太阳黑子示例",
    ),
    DemoScenario(
        scenario_id="synthetic",
        title="Synthetic mixed-structure bundle",
        title_zh="合成混合结构 bundle",
        summary="The fastest smoke test: mixed trend, seasonality, and regime changes that prove the whole report + handoff flow works.",
        summary_zh="最快的 smoke test：混合趋势、周期和 regime 切换，用来证明整条 report + handoff 流程是通的。",
        one_liner="Best for first success, CI, and README screenshots.",
        one_liner_zh="最适合第一次成功、CI 和 README 截图。",
        command_hint="python -m tsdataforge demo --scenario synthetic --output demo_bundle",
        channels=("value",),
        tags=("smoke-test", "mixed-structure", "README"),
    ),
    DemoScenario(
        scenario_id="icu_vitals",
        title="ICU bedside vitals handoff",
        title_zh="ICU 床旁生命体征交接",
        summary="Reality-shaped synthetic bedside vitals with desaturation windows, monitoring gaps, and shift handoff semantics.",
        summary_zh="真实感合成的床旁生命体征数据，包含脱氧窗口、监护缺口和交班语义。",
        one_liner="Best synthetic medicine-facing demo when a biomedical-shaped workflow is needed without shipping external patient data.",
        one_liner_zh="当你需要医学形态工作流、但不想附带外部病人数据时，这是最合适的合成 demo。",
        command_hint="python -m tsdataforge demo --scenario icu_vitals --output icu_bundle",
        channels=("heart_rate", "map", "spo2", "resp_rate"),
        tags=("synthetic", "medicine", "multivariate", "missing", "events"),
    ),
    DemoScenario(
        scenario_id="macro_regime",
        title="Inflation / unemployment / rates regime bundle",
        title_zh="通胀 / 失业 / 利率 regime bundle",
        summary="Reality-shaped synthetic monthly indicators with shock windows and regime changes that make task routing easy to explain.",
        summary_zh="真实感合成的宏观月度指标，包含冲击窗口和 regime 切换，便于解释任务路由。",
        one_liner="Best synthetic economics-facing demo when you want a clean explanatory bundle without external dependencies.",
        one_liner_zh="如果你想要一个干净、易解释、无外部依赖的经济学 demo，这是最合适的合成 bundle。",
        command_hint="python -m tsdataforge demo --scenario macro_regime --output macro_bundle",
        channels=("inflation_yoy", "unemployment", "policy_rate"),
        tags=("synthetic", "economics", "regime", "monthly", "multivariate"),
    ),
    DemoScenario(
        scenario_id="factory_sensor",
        title="Factory sensor drift handoff",
        title_zh="工厂传感器漂移交接",
        summary="Reality-shaped synthetic temperature, vibration, current, and throughput with drift, bursts, and maintenance windows for engineering demos.",
        summary_zh="真实感合成的温度、振动、电流和产量序列，包含漂移、尖峰和维护窗口。",
        one_liner="Best synthetic engineering demo for anomaly, drift, and maintenance narratives.",
        one_liner_zh="最适合做 anomaly、drift 和 maintenance 叙事的工程合成 demo。",
        command_hint="python -m tsdataforge demo --scenario factory_sensor --output factory_bundle",
        channels=("temperature", "vibration", "current", "throughput"),
        tags=("synthetic", "engineering", "drift", "bursty", "multivariate"),
    ),
)


def demo_scenario_catalog(*, language: str = "en") -> tuple[DemoScenario, ...]:
    if language.startswith("zh"):
        localized: list[DemoScenario] = []
        for item in _CATALOG:
            localized.append(
                DemoScenario(
                    scenario_id=item.scenario_id,
                    title=item.title_zh,
                    title_zh=item.title_zh,
                    summary=item.summary_zh,
                    summary_zh=item.summary_zh,
                    one_liner=item.one_liner_zh,
                    one_liner_zh=item.one_liner_zh,
                    command_hint=item.command_hint,
                    channels=item.channels,
                    tags=item.tags,
                    is_real_public_data=item.is_real_public_data,
                    source_label=item.source_label_zh or item.source_label,
                    source_label_zh=item.source_label_zh or item.source_label,
                )
            )
        return tuple(localized)
    return _CATALOG


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(None if seed is None else int(seed))


def _apply_missing_blocks(values: np.ndarray, rng: np.random.Generator, *, n_blocks: int, max_frac: float = 0.08) -> np.ndarray:
    out = np.array(values, dtype=float, copy=True)
    n_steps = out.shape[0]
    for _ in range(max(0, int(n_blocks))):
        ch = int(rng.integers(0, out.shape[1]))
        width = max(2, int(rng.integers(2, max(3, int(round(n_steps * max_frac))))))
        start = int(rng.integers(0, max(1, n_steps - width)))
        out[start : start + width, ch] = np.nan
    return out


def _load_npz(name: str) -> dict[str, np.ndarray]:
    path = DATA_DIR / name
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _window_starts(total: int, length: int, n_series: int) -> np.ndarray:
    if total <= length:
        return np.zeros((max(1, n_series),), dtype=int)
    max_start = total - length
    if n_series <= 1:
        return np.array([max_start // 2], dtype=int)
    starts = np.linspace(0, max_start, num=n_series)
    return np.asarray(np.round(starts), dtype=int)


def _from_real_windows(
    values: np.ndarray,
    time: np.ndarray,
    *,
    n_series: int,
    length: int,
    dataset_id: str,
    scenario: str,
    channel_names: list[str],
    tags: list[str],
    source_label: str,
    entity_prefix: str,
) -> SeriesDataset:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    tt = np.asarray(time, dtype=float)
    length = int(max(32, min(int(length), arr.shape[0])))
    starts = _window_starts(arr.shape[0], length, max(1, int(n_series)))
    windows: list[np.ndarray] = []
    times: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    for i, start in enumerate(starts):
        end = min(arr.shape[0], int(start) + length)
        seg = np.asarray(arr[start:end], dtype=float)
        tseg = np.asarray(tt[start:end] - tt[start], dtype=float)
        windows.append(seg)
        times.append(tseg)
        meta.append(
            {
                "structure_id": scenario,
                "tags": list(tags),
                "scenario": scenario,
                "entity_id": f"{entity_prefix}-{i:03d}",
                "source": source_label,
                "window_start": int(start),
                "window_end": int(end),
            }
        )
    return SeriesDataset.from_arrays(windows, times, meta=meta, dataset_id=dataset_id, channel_names=channel_names)


def _ecg_public_dataset(n_series: int, length: int) -> SeriesDataset:
    payload = _load_npz("ecg_public.npz")
    signal = np.asarray(payload["signal"], dtype=float)
    fs = float(np.asarray(payload["fs"]).reshape(-1)[0])
    time = np.arange(signal.shape[0], dtype=float) / fs
    return _from_real_windows(
        signal,
        time,
        n_series=max(8, int(n_series)),
        length=max(256, int(length)),
        dataset_id="ecg_public_demo",
        scenario="ecg_public",
        channel_names=["ecg_mv"],
        tags=["external", "public-data", "medicine", "biosignal", "bursty", "heavy_tail"],
        source_label="SciPy electrocardiogram / MIT-BIH record 208 excerpt",
        entity_prefix="ecg-window",
    )


def _macro_public_dataset(n_series: int, length: int) -> SeriesDataset:
    payload = _load_npz("macro_public.npz")
    values = np.asarray(payload["values"], dtype=float)
    time = np.asarray(payload["time"], dtype=float)
    channels = [str(x) for x in np.asarray(payload["channel_names"]).tolist()]
    return _from_real_windows(
        values,
        time,
        n_series=max(6, int(n_series)),
        length=max(40, min(int(length), values.shape[0])),
        dataset_id="macro_public_demo",
        scenario="macro_public",
        channel_names=channels,
        tags=["external", "public-data", "economics", "multivariate", "regime", "trend"],
        source_label="statsmodels macrodata sample",
        entity_prefix="macro-window",
    )


def _climate_public_dataset(n_series: int, length: int) -> SeriesDataset:
    payload = _load_npz("climate_co2_public.npz")
    values = np.asarray(payload["values"], dtype=float)
    time = np.asarray(payload["time"], dtype=float)
    channels = [str(x) for x in np.asarray(payload["channel_names"]).tolist()]
    return _from_real_windows(
        values,
        time,
        n_series=max(8, int(n_series)),
        length=max(96, min(int(length), values.shape[0])),
        dataset_id="climate_public_demo",
        scenario="climate_public",
        channel_names=channels,
        tags=["external", "public-data", "climate", "seasonal", "missing", "trend"],
        source_label="statsmodels CO2 sample / Mauna Loa weekly atmospheric CO₂",
        entity_prefix="climate-window",
    )


def _sunspots_public_dataset(n_series: int, length: int) -> SeriesDataset:
    payload = _load_npz("sunspots_public.npz")
    values = np.asarray(payload["values"], dtype=float)
    time = np.asarray(payload["time"], dtype=float)
    channels = [str(x) for x in np.asarray(payload["channel_names"]).tolist()]
    return _from_real_windows(
        values,
        time,
        n_series=max(8, int(n_series)),
        length=max(64, min(int(length), values.shape[0])),
        dataset_id="sunspots_public_demo",
        scenario="sunspots_public",
        channel_names=channels,
        tags=["external", "public-data", "physical-science", "seasonal", "trend"],
        source_label="statsmodels sunspots sample",
        entity_prefix="sunspots-window",
    )


def _icu_vitals_dataset(n_series: int, length: int, seed: int) -> SeriesDataset:
    rng = _rng(seed)
    t = np.arange(int(length), dtype=float)
    values: list[np.ndarray] = []
    times: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    for i in range(int(n_series)):
        phase = rng.uniform(0, 2 * np.pi)
        hr = 82.0 + 5.5 * np.sin(2 * np.pi * t / 60.0 + phase) + rng.normal(0, 1.6, size=len(t))
        map_ = 78.0 + 1.8 * np.sin(2 * np.pi * t / 90.0 + phase / 2) + rng.normal(0, 1.2, size=len(t))
        spo2 = 97.5 + 0.3 * np.sin(2 * np.pi * t / 70.0 + phase / 3) + rng.normal(0, 0.15, size=len(t))
        rr = 16.0 + 1.2 * np.sin(2 * np.pi * t / 45.0 + phase / 4) + rng.normal(0, 0.5, size=len(t))
        n_events = 1 + int(rng.integers(1, 3))
        for _ in range(n_events):
            center = int(rng.integers(max(10, len(t) // 8), max(11, len(t) - len(t) // 8)))
            width = float(rng.uniform(4.0, 10.0))
            envelope = np.exp(-0.5 * ((t - center) / width) ** 2)
            hr += rng.uniform(10.0, 18.0) * envelope
            map_ -= rng.uniform(5.0, 10.0) * envelope
            spo2 -= rng.uniform(2.0, 5.0) * envelope
            rr += rng.uniform(2.0, 4.0) * envelope
        arr = np.stack((hr, map_, np.clip(spo2, 85.0, 100.0), rr), axis=1)
        arr = _apply_missing_blocks(arr, rng, n_blocks=2)
        values.append(arr)
        times.append(t)
        meta.append(
            {
                "structure_id": "icu_vitals",
                "tags": ["external", "medicine", "multivariate", "missing", "events"],
                "subject_id": f"icu-bed-{i:03d}",
                "scenario": "icu_vitals",
                "source": "synthetic-reality-shaped",
            }
        )
    return SeriesDataset.from_arrays(values, times, meta=meta, dataset_id="icu_vitals_demo", channel_names=list(["heart_rate", "map", "spo2", "resp_rate"]))


def _macro_regime_dataset(n_series: int, length: int, seed: int) -> SeriesDataset:
    rng = _rng(seed)
    t = np.arange(int(length), dtype=float)
    values: list[np.ndarray] = []
    times: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    for i in range(int(n_series)):
        base_infl = 2.0 + 0.15 * np.sin(2 * np.pi * t / 12.0 + rng.uniform(0, 2 * np.pi))
        base_unemp = 4.8 + 0.2 * np.sin(2 * np.pi * t / 18.0 + rng.uniform(0, 2 * np.pi))
        base_rate = 2.5 + 0.12 * np.sin(2 * np.pi * t / 24.0 + rng.uniform(0, 2 * np.pi))
        inflation = base_infl + rng.normal(0, 0.08, size=len(t))
        unemployment = base_unemp + rng.normal(0, 0.06, size=len(t))
        policy_rate = base_rate + rng.normal(0, 0.05, size=len(t))
        shock_start = int(rng.integers(max(12, len(t) // 4), max(13, len(t) // 2)))
        shock_end = min(len(t), shock_start + int(rng.integers(8, 18)))
        inflation[shock_start:shock_end] += rng.uniform(1.0, 2.2)
        unemployment[shock_start:shock_end] += rng.uniform(0.4, 1.0)
        policy_rate[shock_start:shock_end] += rng.uniform(0.8, 1.8)
        cooldown = slice(shock_end, min(len(t), shock_end + int(rng.integers(6, 14))))
        unemployment[cooldown] += rng.uniform(0.2, 0.8)
        arr = np.stack((inflation, unemployment, np.maximum(policy_rate, 0.0)), axis=1)
        values.append(arr)
        times.append(t)
        meta.append(
            {
                "structure_id": "macro_regime",
                "tags": ["external", "economics", "multivariate", "regime"],
                "panel_id": f"macro-panel-{i:03d}",
                "scenario": "macro_regime",
                "source": "synthetic-reality-shaped",
            }
        )
    return SeriesDataset.from_arrays(values, times, meta=meta, dataset_id="macro_regime_demo", channel_names=list(["inflation_yoy", "unemployment", "policy_rate"]))


def _factory_sensor_dataset(n_series: int, length: int, seed: int) -> SeriesDataset:
    rng = _rng(seed)
    t = np.arange(int(length), dtype=float)
    values: list[np.ndarray] = []
    times: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    for i in range(int(n_series)):
        drift = np.linspace(0.0, rng.uniform(2.0, 5.0), len(t))
        temp = 67.0 + drift + 0.6 * np.sin(2 * np.pi * t / 36.0) + rng.normal(0, 0.35, size=len(t))
        vibration = 1.1 + 0.05 * t / max(1, len(t)) + rng.normal(0, 0.05, size=len(t))
        current = 12.0 + 0.12 * np.sin(2 * np.pi * t / 24.0 + rng.uniform(0, 2 * np.pi)) + rng.normal(0, 0.12, size=len(t))
        throughput = 100.0 - 0.4 * drift + rng.normal(0, 0.8, size=len(t))
        for _ in range(1 + int(rng.integers(1, 3))):
            center = int(rng.integers(max(8, len(t) // 8), max(9, len(t) - len(t) // 10)))
            width = float(rng.uniform(2.0, 6.0))
            envelope = np.exp(-0.5 * ((t - center) / width) ** 2)
            vibration += rng.uniform(0.25, 0.45) * envelope
            current += rng.uniform(0.5, 0.9) * envelope
            throughput -= rng.uniform(3.0, 7.0) * envelope
        maintenance = int(rng.integers(max(12, len(t) // 3), max(13, len(t) - len(t) // 4)))
        throughput[maintenance : maintenance + 3] = np.nan
        temp[maintenance : maintenance + 3] -= 1.5
        arr = np.stack((temp, np.maximum(vibration, 0.0), np.maximum(current, 0.0), throughput), axis=1)
        values.append(arr)
        times.append(t)
        meta.append(
            {
                "structure_id": "factory_sensor",
                "tags": ["external", "engineering", "multivariate", "drift", "bursty"],
                "unit_id": f"line-{i:03d}",
                "scenario": "factory_sensor",
                "source": "synthetic-reality-shaped",
            }
        )
    return SeriesDataset.from_arrays(values, times, meta=meta, dataset_id="factory_sensor_demo", channel_names=list(["temperature", "vibration", "current", "throughput"]))


def build_demo_dataset(
    *,
    scenario: str = "synthetic",
    n_series: int = 24,
    length: int = 192,
    seed: int = 0,
) -> SeriesDataset:
    from .datasets.series_dataset import generate_series_dataset

    scenario = str(scenario).lower().strip()
    if scenario == "synthetic":
        dataset = generate_series_dataset(
            structures=["trend_seasonal_noise", "regime_switch"],
            n_series=int(n_series),
            length=int(length),
            seed=int(seed),
            sampling="balanced",
        )
        dataset.dataset_id = "synthetic_demo"
        return dataset
    if scenario == "ecg_public":
        return _ecg_public_dataset(n_series=max(8, int(n_series)), length=max(256, int(length)))
    if scenario == "macro_public":
        return _macro_public_dataset(n_series=max(6, int(n_series)), length=max(40, int(length)))
    if scenario == "climate_public":
        return _climate_public_dataset(n_series=max(8, int(n_series)), length=max(96, int(length)))
    if scenario == "sunspots_public":
        return _sunspots_public_dataset(n_series=max(8, int(n_series)), length=max(64, int(length)))
    if scenario == "icu_vitals":
        return _icu_vitals_dataset(n_series=max(8, int(n_series)), length=max(96, int(length)), seed=int(seed))
    if scenario == "macro_regime":
        return _macro_regime_dataset(n_series=max(6, int(n_series)), length=max(84, int(length)), seed=int(seed))
    if scenario == "factory_sensor":
        return _factory_sensor_dataset(n_series=max(8, int(n_series)), length=max(120, int(length)), seed=int(seed))
    valid = ", ".join(item.scenario_id for item in _CATALOG)
    raise ValueError(f"Unknown demo scenario: {scenario!r}. Valid scenarios: {valid}")


__all__ = ["DemoScenario", "demo_scenario_catalog", "build_demo_dataset"]
