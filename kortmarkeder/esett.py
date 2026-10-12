"""Klient mot eSett Open Data API (https://api.opendata.esett.com).

eSett er oppgjørsansvarlig for ubalanser i Norden. API-et er åpent og krever ingen nøkkel.
Vi bruker to endepunkter, begge med 15-minutters oppløsning per budområde (MBA):

    EXP14/Prices                – ubalansepris, opp-/nedreguleringspris, hovedretning for
                                  regulering og differansen ubalansepris − spotpris
    EXP13/ImbalancePowerVolume  – ubalansevolum (MWh), fordelt på kjøp og salg

API-et svarer raskt på én måned om gangen, så lengre perioder deles opp i månedsbiter.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
import requests

BASE_URL = "https://api.opendata.esett.com"

# eSett sine MBA-koder (merk: understrek, ikke bindestrek, for NO1–NO4)
MBA_CODES = {
    "NO1": "10YNO_1________2",
    "NO2": "10YNO_2________T",
    "NO3": "10YNO_3________J",
    "NO4": "10YNO_4________9",
    "NO5": "10Y1001A1001A48H",
}

PRICE_COLS = [
    "imblSalesPrice", "imblPurchasePrice", "upRegPrice", "downRegPrice",
    "mainDirRegPowerPerMBA", "imblSpotDifferencePrice", "upRegPriceFrrA",
    "downRegPriceFrrA", "valueOfAvoidedActivation", "incentivisingComponent",
]
VOLUME_COLS = ["imbalance", "imbalancePurchase", "imbalanceSales"]
ALL_COLS = ["timestamp_utc", "area"] + PRICE_COLS + VOLUME_COLS

ENDPOINTS = {
    "prices": ("/EXP14/Prices", PRICE_COLS),
    "volumes": ("/EXP13/ImbalancePowerVolume", VOLUME_COLS),
}


def _iso(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _month_chunks(start: pd.Timestamp, end: pd.Timestamp):
    """Del [start, end) i biter som ikke krysser månedsgrenser."""
    cur = start
    while cur < end:
        nxt = (cur + pd.offsets.MonthBegin(1)).normalize()
        if nxt <= cur:
            nxt = cur + pd.offsets.MonthBegin(1)
        yield cur, min(nxt, end)
        cur = min(nxt, end)


def _get(session: requests.Session, path: str, params: dict, retries: int = 3) -> list[dict]:
    for attempt in range(retries):
        try:
            r = session.get(BASE_URL + path, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return []


def parse_records(records: list[dict], cols: list[str], area: str) -> pd.DataFrame:
    """Gjør om JSON-svaret fra eSett til en tabell med UTC-tidsstempel som nøkkel."""
    if not records:
        return pd.DataFrame(columns=["timestamp_utc", "area"] + cols)
    df = pd.DataFrame.from_records(records)
    out = pd.DataFrame({"timestamp_utc": pd.to_datetime(df["timestampUTC"], utc=True), "area": area})
    for c in cols:
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df else float("nan")
    return out


def fetch(area: str, start, end, session: requests.Session | None = None,
          verbose: bool = True) -> pd.DataFrame:
    """Hent priser og volumer for ett budområde i perioden [start, end) (UTC).

    Returnerer én rad per 15-minutters avregningsperiode (MTU) med alle kolonner i ALL_COLS.
    """
    session = session or requests.Session()
    start, end = to_utc(start), to_utc(end)
    mba = MBA_CODES[area]

    parts: dict[str, list[pd.DataFrame]] = {k: [] for k in ENDPOINTS}
    for a, b in _month_chunks(start, end):
        for key, (path, cols) in ENDPOINTS.items():
            recs = _get(session, path, {"start": _iso(a), "end": _iso(b), "mba": mba})
            parts[key].append(parse_records(recs, cols, area))
        if verbose:
            print(f"  {area} {a:%Y-%m-%d} → {b:%Y-%m-%d}: {len(parts['prices'][-1])} rader")

    prices = pd.concat(parts["prices"], ignore_index=True)
    volumes = pd.concat(parts["volumes"], ignore_index=True)
    df = prices.merge(volumes, on=["timestamp_utc", "area"], how="outer")
    df = df.drop_duplicates(["timestamp_utc", "area"], keep="last")
    return df.sort_values("timestamp_utc")[ALL_COLS].reset_index(drop=True)


def to_utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def now_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc)).floor("15min")
