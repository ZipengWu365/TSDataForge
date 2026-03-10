from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

import numpy as np

from .external import wrap_external_series
from ..series import GeneratedSeries


DEFAULT_USER_AGENT = "tsdataforge/0.3.4"
COMMON_FRED_SERIES: dict[str, str] = {
    "gold_usd_per_oz": "GOLDPMGBD228NLBM",
    "wti_oil_usd_per_barrel": "DCOILWTICO",
    "brent_oil_usd_per_barrel": "DCOILBRENTEU",
}


@dataclass(frozen=True)
class LiveProviderHint:
    provider: str
    env_vars: tuple[str, ...]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {"provider": self.provider, "env_vars": list(self.env_vars), "notes": self.notes}


LIVE_PROVIDER_HINTS: tuple[LiveProviderHint, ...] = (
    LiveProviderHint(
        provider="github",
        env_vars=("GITHUB_TOKEN",),
        notes="Public REST stargazer history can be fetched without auth for public repositories, but a token is strongly recommended for larger repos or GraphQL mode.",
    ),
    LiveProviderHint(
        provider="fred",
        env_vars=("FRED_API_KEY",),
        notes="FRED observations require an API key. Daily gold and oil series are convenient for macro case studies.",
    ),
    LiveProviderHint(
        provider="coingecko",
        env_vars=("COINGECKO_API_KEY", "COINGECKO_BASE_URL"),
        notes="CoinGecko market_chart data is convenient for crypto price examples. Supply an API key when your plan requires it.",
    ),
)


def _request_json(url: str, *, headers: dict[str, str] | None = None, params: dict[str, Any] | None = None, payload: bytes | None = None, timeout: int = 30) -> Any:
    if params:
        sep = "&" if "?" in url else "?"
        url = url + sep + urlencode({k: v for k, v in params.items() if v is not None})
    req = Request(url, data=payload, headers={**(headers or {}), "User-Agent": DEFAULT_USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310 - user explicitly requested live integrations
        raw = resp.read().decode("utf-8")
    return json.loads(raw)



def _parse_optional_date(value: str | date | datetime | None) -> np.datetime64 | None:
    if value is None:
        return None
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[D]")
    if isinstance(value, datetime):
        return np.datetime64(value.date(), "D")
    if isinstance(value, date):
        return np.datetime64(value, "D")
    text = str(value).strip()
    if not text:
        return None
    return np.datetime64(text[:10], "D")



def _daily_index(date_strings: Iterable[str]) -> tuple[np.ndarray, list[str]]:
    dates = np.array([np.datetime64(str(item)[:10], "D") for item in date_strings], dtype="datetime64[D]")
    if dates.size == 0:
        return np.zeros((0,), dtype=float), []
    unique = np.unique(dates)
    t = (unique - unique[0]).astype("timedelta64[D]").astype(float)
    return t, [str(x) for x in unique]



def _attach_metadata(series: GeneratedSeries, *, provider: str, source_url: str, date_strings: list[str], extra_states: dict[str, Any] | None = None) -> GeneratedSeries:
    if series.trace is not None:
        series.trace.states[f"external/{provider}/date_strings"] = list(date_strings)
        series.trace.states[f"external/{provider}/source_url"] = str(source_url)
        if extra_states:
            for key, value in extra_states.items():
                series.trace.states[key] = value
    return series



def fetch_fred_series(
    series_id: str,
    *,
    api_key: str | None = None,
    observation_start: str | date | datetime | None = None,
    observation_end: str | date | datetime | None = None,
    frequency: str | None = None,
    aggregation_method: str | None = None,
    units: str | None = None,
    name: str | None = None,
) -> GeneratedSeries:
    """Fetch one FRED series and wrap it as a GeneratedSeries.

    This is primarily meant for real-world case studies such as gold and oil
    price analysis, not as a heavy macro-data client.
    """

    key = api_key or os.getenv("FRED_API_KEY")
    if not key:
        raise ValueError("FRED API key not provided. Set api_key=... or FRED_API_KEY.")
    params = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
    }
    if observation_start is not None:
        params["observation_start"] = str(_parse_optional_date(observation_start))
    if observation_end is not None:
        params["observation_end"] = str(_parse_optional_date(observation_end))
    if frequency is not None:
        params["frequency"] = str(frequency)
    if aggregation_method is not None:
        params["aggregation_method"] = str(aggregation_method)
    if units is not None:
        params["units"] = str(units)

    url = "https://api.stlouisfed.org/fred/series/observations?" + urlencode(params)
    payload = _request_json(url)
    observations = payload.get("observations", [])
    date_strings: list[str] = []
    values: list[float] = []
    for item in observations:
        value = str(item.get("value", ""))
        if value == ".":
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        date_strings.append(str(item.get("date", ""))[:10])
        values.append(v)
    t, dates = _daily_index(date_strings)
    series = wrap_external_series(
        np.asarray(values, dtype=float),
        t,
        name=name or f"FRED:{series_id}",
        tags=("external", "fred", series_id.lower()),
        meta={"provider": "fred", "series_id": series_id},
    )
    return _attach_metadata(
        series,
        provider="fred",
        source_url=url,
        date_strings=dates,
        extra_states={f"external/fred/series_id": series_id},
    )



def fetch_coingecko_market_chart(
    coin_id: str,
    *,
    vs_currency: str = "usd",
    days: int | str = 90,
    interval: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    name: str | None = None,
) -> GeneratedSeries:
    """Fetch CoinGecko market-chart data and wrap the price series.

    The connector stores daily price as the observed values and attaches market
    cap / volume side channels to the trace when they are present.
    """

    key = api_key or os.getenv("COINGECKO_API_KEY")
    root = base_url or os.getenv("COINGECKO_BASE_URL") or ("https://pro-api.coingecko.com/api/v3" if key else "https://api.coingecko.com/api/v3")
    params: dict[str, Any] = {"vs_currency": vs_currency, "days": days}
    if interval is not None:
        params["interval"] = interval
    url = f"{root.rstrip('/')}/coins/{quote(coin_id)}/market_chart?" + urlencode(params)
    headers = {}
    if key:
        headers["x-cg-pro-api-key"] = key
    payload = _request_json(url, headers=headers)

    price_rows = payload.get("prices", [])
    cap_rows = payload.get("market_caps", [])
    vol_rows = payload.get("total_volumes", [])
    daily_price: dict[str, float] = {}
    daily_cap: dict[str, float] = {}
    daily_vol: dict[str, float] = {}
    for ts_ms, value in price_rows:
        day = str(np.datetime64(int(ts_ms), "ms").astype("datetime64[D]"))
        daily_price[day] = float(value)
    for ts_ms, value in cap_rows:
        day = str(np.datetime64(int(ts_ms), "ms").astype("datetime64[D]"))
        daily_cap[day] = float(value)
    for ts_ms, value in vol_rows:
        day = str(np.datetime64(int(ts_ms), "ms").astype("datetime64[D]"))
        daily_vol[day] = float(value)

    dates = sorted(daily_price)
    values = np.asarray([daily_price[d] for d in dates], dtype=float)
    t, ordered_dates = _daily_index(dates)
    series = wrap_external_series(
        values,
        t,
        name=name or f"CoinGecko:{coin_id}",
        tags=("external", "coingecko", coin_id.lower()),
        meta={"provider": "coingecko", "coin_id": coin_id, "vs_currency": vs_currency},
    )
    return _attach_metadata(
        series,
        provider="coingecko",
        source_url=url,
        date_strings=ordered_dates,
        extra_states={
            "external/coingecko/market_caps": np.asarray([daily_cap.get(d, np.nan) for d in ordered_dates], dtype=float),
            "external/coingecko/total_volumes": np.asarray([daily_vol.get(d, np.nan) for d in ordered_dates], dtype=float),
            "external/coingecko/coin_id": coin_id,
            "external/coingecko/vs_currency": vs_currency,
        },
    )



def _github_headers(token: str | None, *, rest: bool = True) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github.star+json" if rest else "application/json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers



def _aggregate_daily_counts(date_strings: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    daily = Counter(str(item)[:10] for item in date_strings)
    dates = sorted(daily)
    new = np.asarray([daily[d] for d in dates], dtype=float)
    cumulative = np.cumsum(new)
    return dates, new, cumulative



def _fetch_github_star_dates_rest(owner: str, repo: str, *, token: str | None = None, max_pages: int = 100, per_page: int = 100) -> list[str]:
    star_dates: list[str] = []
    for page in range(1, max(1, int(max_pages)) + 1):
        url = f"https://api.github.com/repos/{quote(owner)}/{quote(repo)}/stargazers"
        payload = _request_json(url, headers=_github_headers(token, rest=True), params={"per_page": int(per_page), "page": page})
        if not isinstance(payload, list) or not payload:
            break
        batch = [str(item.get("starred_at", ""))[:10] for item in payload if item.get("starred_at")]
        star_dates.extend(batch)
        if len(payload) < int(per_page):
            break
    return star_dates



def _fetch_github_star_dates_graphql(
    owner: str,
    repo: str,
    *,
    token: str,
    since: str | date | datetime | None = None,
    until: str | date | datetime | None = None,
    max_pages: int = 100,
) -> list[str]:
    since_dt = _parse_optional_date(since)
    until_dt = _parse_optional_date(until)
    url = "https://api.github.com/graphql"
    query = """
    query($owner: String!, $repo: String!, $cursor: String) {
      repository(owner: $owner, name: $repo) {
        stargazers(first: 100, after: $cursor, orderBy: {field: STARRED_AT, direction: DESC}) {
          edges { starredAt }
          pageInfo { hasNextPage endCursor }
        }
      }
    }
    """
    cursor = None
    dates: list[str] = []
    for _ in range(max(1, int(max_pages))):
        variables = {"owner": owner, "repo": repo, "cursor": cursor}
        payload = _request_json(
            url,
            headers={**_github_headers(token, rest=False), "Content-Type": "application/json"},
            payload=json.dumps({"query": query, "variables": variables}).encode("utf-8"),
        )
        repo_data = ((payload or {}).get("data") or {}).get("repository") or {}
        stars = repo_data.get("stargazers") or {}
        edges = stars.get("edges") or []
        if not edges:
            break
        stop = False
        for edge in edges:
            ts = str(edge.get("starredAt", ""))
            if not ts:
                continue
            day = np.datetime64(ts[:10], "D")
            if until_dt is not None and day > until_dt:
                continue
            if since_dt is not None and day < since_dt:
                stop = True
                break
            dates.append(str(day))
        page_info = stars.get("pageInfo") or {}
        if stop or not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor")
        if not cursor:
            break
    return sorted(dates)



def fetch_github_stars_series(
    owner: str,
    repo: str,
    *,
    token: str | None = None,
    mode: str = "graphql",
    since: str | date | datetime | None = None,
    until: str | date | datetime | None = None,
    max_pages: int = 100,
    per_page: int = 100,
    name: str | None = None,
) -> GeneratedSeries:
    """Fetch GitHub repository star history and wrap it as a GeneratedSeries.

    Parameters
    ----------
    mode:
        ``"graphql"`` is the recommended path for large, fast-growing repos
        because it can request stargazers in descending STARRED_AT order and stop
        once the requested window is reached. ``"rest"`` is simpler and works
        without auth for public repositories, but may require many requests for
        large repos.
    """

    auth_token = token or os.getenv("GITHUB_TOKEN")
    mode_norm = (mode or "graphql").lower()
    if mode_norm == "graphql":
        if not auth_token:
            raise ValueError("GitHub GraphQL mode requires a token. Set token=... or GITHUB_TOKEN.")
        star_dates = _fetch_github_star_dates_graphql(
            owner,
            repo,
            token=auth_token,
            since=since,
            until=until,
            max_pages=max_pages,
        )
    elif mode_norm == "rest":
        star_dates = _fetch_github_star_dates_rest(owner, repo, token=auth_token, max_pages=max_pages, per_page=per_page)
        since_dt = _parse_optional_date(since)
        until_dt = _parse_optional_date(until)
        filtered: list[str] = []
        for item in star_dates:
            day = np.datetime64(item[:10], "D")
            if since_dt is not None and day < since_dt:
                continue
            if until_dt is not None and day > until_dt:
                continue
            filtered.append(str(day))
        star_dates = sorted(filtered)
    else:
        raise ValueError("mode must be 'graphql' or 'rest'")

    dates, daily_new, cumulative = _aggregate_daily_counts(star_dates)
    t, ordered_dates = _daily_index(dates)
    series = wrap_external_series(
        cumulative,
        t,
        name=name or f"GitHub stars: {owner}/{repo}",
        tags=("external", "github", "stargazers", owner.lower(), repo.lower()),
        meta={"provider": "github", "owner": owner, "repo": repo, "mode": mode_norm},
    )
    return _attach_metadata(
        series,
        provider="github",
        source_url=f"https://github.com/{owner}/{repo}",
        date_strings=ordered_dates,
        extra_states={
            "external/github/owner": owner,
            "external/github/repo": repo,
            "external/github/mode": mode_norm,
            "external/github/daily_new": daily_new,
            "external/github/daily_new_stars": daily_new,
            "external/github/cumulative": cumulative,
            "external/github/cumulative_stars": cumulative,
        },
    )
