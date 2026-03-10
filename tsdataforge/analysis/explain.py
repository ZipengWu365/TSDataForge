from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .describe import SeriesDescription


@dataclass
class SeriesExplanation:
    """Human-readable explanation for a `SeriesDescription`.

    This is deliberately *interpretable* and grounded in the computed
    statistics/scores. It is not a statistical guarantee.
    """

    headline: str
    bullets: list[str] = field(default_factory=list)
    tag_explanations: dict[str, str] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def explain_series(desc: SeriesDescription) -> SeriesExplanation:
    tags = list(desc.inferred_tags)
    scores = dict(desc.scores or {})

    bullets: list[str] = []
    tag_expl: dict[str, str] = {}
    evidence: dict[str, Any] = {}

    parts: list[str] = []
    if "multivariate" in tags:
        parts.append(f"{desc.n_channels}-channel multivariate")
    else:
        parts.append("univariate")
    if "irregular_sampling" in tags:
        parts.append("irregularly sampled")
    else:
        parts.append("regularly sampled")
    if "missing" in tags:
        parts.append("with missingness")
    headline = " ".join(parts).strip().capitalize() + "."

    if "missing" in tags:
        mr = float(scores.get("missing_rate", desc.missing_rate))
        tag_expl["missing"] = f"Missing values detected (missing_rate={mr:.2%})."
        bullets.append(tag_expl["missing"])
        evidence["missing"] = {"missing_rate": mr}

    if "irregular_sampling" in tags:
        dt_cv = float(scores.get("dt_cv", desc.dt_cv))
        tag_expl["irregular_sampling"] = (
            "Sampling intervals vary noticeably "
            f"(dt coefficient-of-variation={dt_cv:.3f}); treat this as an irregularly sampled series."
        )
        bullets.append(tag_expl["irregular_sampling"])
        evidence["irregular_sampling"] = {"dt_mean": desc.dt_mean, "dt_cv": dt_cv}

    if "trend" in tags:
        r2 = float(scores.get("trend_r2_median", float("nan")))
        slope = float(scores.get("trend_slope_mean_abs", float("nan")))
        tag_expl["trend"] = (
            "A linear trend explains a non-trivial fraction of variance "
            f"(median R^2={r2:.2f}); typical absolute slope={slope:.4g} per time unit."
        )
        bullets.append(tag_expl["trend"])
        evidence["trend"] = {"trend_r2_median": r2, "trend_slope_mean_abs": slope}

    if "seasonal" in tags:
        period = float(scores.get("dominant_period", float("nan")))
        ratio = float(scores.get("seasonality_power_ratio", float("nan")))
        tag_expl["seasonal"] = (
            "A dominant periodic component is present "
            f"(period={period:.3g}, spectral power_ratio={ratio:.2f})."
        )
        bullets.append(tag_expl["seasonal"])
        evidence["seasonal"] = {"dominant_period": period, "power_ratio": ratio}

    if "random_walk_like" in tags:
        a1 = float(scores.get("acf1_median", float("nan")))
        tag_expl["random_walk_like"] = (
            "Very high lag-1 autocorrelation suggests a random-walk-like / "
            f"integrated process (acf1={a1:.3f})."
        )
        bullets.append(tag_expl["random_walk_like"])
        evidence["random_walk_like"] = {"acf1_median": a1}
    elif "ar1_like" in tags:
        a1 = float(scores.get("acf1_median", float("nan")))
        tag_expl["ar1_like"] = (
            "Moderate lag-1 autocorrelation is consistent with AR(1)-like "
            f"colored noise (acf1={a1:.3f})."
        )
        bullets.append(tag_expl["ar1_like"])
        evidence["ar1_like"] = {"acf1_median": a1}
    elif "white_noise_like" in tags:
        a1 = float(scores.get("acf1_median", float("nan")))
        tag_expl["white_noise_like"] = (
            "Low lag-1 autocorrelation suggests near-white-noise behavior "
            f"(acf1={a1:.3f})."
        )
        bullets.append(tag_expl["white_noise_like"])
        evidence["white_noise_like"] = {"acf1_median": a1}

    if "bursty" in tags or "heavy_tail" in tags:
        km = float(scores.get("kurtosis_median", float("nan")))
        sm = float(scores.get("spikiness_median", float("nan")))
        tag_expl["bursty"] = (
            "Spiky / heavy-tailed behavior detected "
            f"(median kurtosis={km:.2f}, spikiness={sm:.2%})."
        )
        bullets.append(tag_expl["bursty"])
        evidence["bursty"] = {"kurtosis_median": km, "spikiness_median": sm}

    if "coupled" in tags:
        cc = float(scores.get("cross_corr_mean_abs", float("nan")))
        tag_expl["coupled"] = (
            "Channels show strong average absolute cross-correlation "
            f"(mean|corr|={cc:.2f}), suggesting coupling/shared latent drivers."
        )
        bullets.append(tag_expl["coupled"])
        evidence["coupled"] = {"cross_corr_mean_abs": cc}

    bullets.append(
        "Tags are produced by lightweight, explainable heuristics (not formal hypothesis tests). "
        "Use them as a starting point for modeling or benchmarking."
    )

    return SeriesExplanation(
        headline=headline,
        bullets=bullets,
        tag_explanations=tag_expl,
        evidence=evidence,
    )
