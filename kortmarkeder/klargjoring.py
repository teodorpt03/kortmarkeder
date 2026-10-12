"""Rensing, avledede størrelser og kalendervariabler.

Nøkkelstørrelser (EUR/MWh og MWh per 15 min):
    spot       – day-ahead-pris. Avledes som ubalansepris − imblSpotDifferencePrice
                 (eSett definerer differansen slik; sjekket mot opp/ned-regpris i dataene).
    ubal_pris  – ubalansepris (én-pris-modell: kjøp = salg).
    spread     – ubal_pris − spot: "premien" for å stå i ubalanse i stedet for å handle i spot.
    retning    – hovedretning for regulering: +1 opp (underskudd), −1 ned (overskudd), 0 ingen.
    ubal_mwh   – avregnet ubalansevolum. Positivt henger sammen med nedregulering, dvs.
                 overskudd i systemet (empirisk: snitt +16 MWh når retning = −1).

Markedsdesign-regimer i dataene (viktig for modellering):
    R1  < 2025-03-04          Timesoppløsning i praksis for både spot og ubalansepris
    R2  2025-03-04–2025-10-01 Nordisk mFRR-energiaktiveringsmarked (EAM): ubalansepris per 15 min,
                              spot fortsatt per time
    R3  >= 2025-10-01         15-minutters day-ahead (spot) – hele kjeden i kvartersoppløsning
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIME_BREAKS = [
    (pd.Timestamp("2025-03-04", tz="UTC"), "R2: mFRR EAM (15-min ubalanse)"),
    (pd.Timestamp("2025-10-01", tz="UTC"), "R3: 15-min spot"),
]
R3_START = REGIME_BREAKS[-1][0]
TZ = "Europe/Oslo"


def regime(ts: pd.Series) -> pd.Series:
    out = pd.Series("R1: timesoppløsning", index=ts.index)
    for start, name in REGIME_BREAKS:
        out[ts >= start] = name
    return out


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Lang tabell (area, timestamp_utc) → renset tabell med avledede kolonner."""
    df = raw.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.sort_values(["area", "timestamp_utc"]).reset_index(drop=True)

    df["ubal_pris"] = df["imblSalesPrice"]
    df["spot"] = df["imblSalesPrice"] - df["imblSpotDifferencePrice"]
    df["spread"] = df["imblSpotDifferencePrice"]
    df["opp_pris"] = df["upRegPrice"]
    df["ned_pris"] = df["downRegPrice"]
    df["retning"] = df["mainDirRegPowerPerMBA"].astype("Int64")
    df["ubal_mwh"] = df["imbalance"]

    local = df["timestamp_utc"].dt.tz_convert(TZ)
    df["tid_lokal"] = local
    df["dato"] = local.dt.date
    df["time"] = local.dt.hour
    df["kvarter"] = local.dt.minute // 15          # 0..3 innenfor timen
    df["mtu_i_dognet"] = df["time"] * 4 + df["kvarter"]
    df["ukedag"] = local.dt.dayofweek              # 0 = mandag
    df["helg"] = df["ukedag"] >= 5
    df["maaned"] = local.dt.month
    df["regime"] = regime(df["timestamp_utc"])

    keep = ["timestamp_utc", "area", "tid_lokal", "dato", "time", "kvarter", "mtu_i_dognet",
            "ukedag", "helg", "maaned", "regime", "spot", "ubal_pris", "spread", "opp_pris",
            "ned_pris", "retning", "ubal_mwh"]
    return df[keep]


def wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Én kolonne per område, indeksert på UTC-tid og med regelmessig 15-min-grid."""
    w = df.pivot(index="timestamp_utc", columns="area", values=col).sort_index()
    full = pd.date_range(w.index.min(), w.index.max(), freq="15min", tz="UTC")
    return w.reindex(full)


def area_series(df: pd.DataFrame, area: str, start=None) -> pd.DataFrame:
    """Ett område som regelmessig 15-min tidsserie (manglende MTU-er blir NaN)."""
    d = df[df["area"] == area].set_index("timestamp_utc").sort_index()
    if start is not None:
        d = d[d.index >= pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None
              else d.index >= pd.Timestamp(start)]
    full = pd.date_range(d.index.min(), d.index.max(), freq="15min", tz="UTC")
    return d.reindex(full)


def winsor(s: pd.Series, q: float = 0.001) -> pd.Series:
    lo, hi = s.quantile([q, 1 - q])
    return s.clip(lo, hi)


def safe_div(a, b):
    return np.where(b == 0, np.nan, a / np.where(b == 0, 1, b))
