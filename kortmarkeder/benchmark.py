"""Benchmark-modeller for pris og volum i de korte markedene etter spot.

Prognoseoppsett (direkte h-stegs prognose):
    Ved tidspunkt t lages en prognose for MTU τ = t + h, kun med informasjon til og med t.
    h måles i kvarter (MTU): 1 = 15 min, 4 = 1 time, 16 = 4 timer, 96 = ett døgn.
    Antakelse: ubalansedata for MTU t er kjent ved t. I drift kommer data med noe forsinkelse,
    som i praksis betyr at reell horisont = h + publiseringsforsinkelse.

    Spotprisen for τ regnes som kjent når h ≤ 44 (11 timer), fordi day-ahead-auksjonen
    for neste døgn klareres rundt kl. 13 dagen før. For h = 96 brukes den ikke.

Mål (targets):
    spread    – ubalansepris − spot (EUR/MWh). Ubalansepris = spot + spread, og spot er kjent,
                så å varsle spread er det samme som å varsle ubalanseprisen.
    ubal_mwh  – avregnet ubalansevolum (MWh per kvarter)
    retning   – reguleringsretning (+1 opp / −1 ned), klassifisering

Modeller (fra enkle referanser til en første "egen" modell):
    nullmodell   spread = 0 ("spot er forventningsrett"), volum = treningssnitt, retning = vanligst
    persistens   ŷ(τ) = y(t)
    sesongnaiv   ŷ(τ) = y(τ − 96)  (samme kvarter i går)
    klimatologi  snitt av samme kvarter-i-døgnet de siste 28 dagene
    ARX (ridge)  lineær regresjon på egne lagger, sesonglagger, spot og kalender
    GBM          gradient boosting (HistGradientBoosting) på samme forklaringsvariabler

Evaluering: utvidende vindu, ny tilpasning hver måned (ingen fremtidsinformasjon i trening).
Mål: MAE (robust mot pristopper), RMSE og "skill" = 1 − MAE/MAE(persistens).
For retning: treffsikkerhet (accuracy) og Brier-score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import AREAS, klargjoring as k, lagring

RESULTS = lagring.ROOT / "results"
HORIZONS = [1, 4, 16, 96]
TRAIN_START = "2025-03-04"   # fra mFRR EAM: ubalansepris i 15-min oppløsning
TEST_START = "2026-01-01"
SPOT_KNOWN_MAX_H = 44
N_LAGS = 8


def features(a: pd.DataFrame, target: str, h: int) -> pd.DataFrame:
    """Forklaringsvariabler for τ, bygget kun fra data ≤ t = τ − h (pluss kjent spot/kalender)."""
    y = a[target].astype(float)
    X = pd.DataFrame(index=a.index)
    for j in range(N_LAGS):
        X[f"lag_{j}"] = y.shift(h + j)                       # y(t − j)
    X["snitt_4t"] = y.shift(h).rolling(16, min_periods=4).mean()
    X["sesong_1d"] = y.shift(96)                            # y(τ − 96), kjent når h ≤ 96
    X["sesong_7d"] = y.shift(672)
    if target != "retning":
        X["retning_t"] = a["retning"].astype(float).shift(h)
        X["retning_snitt_4t"] = a["retning"].astype(float).shift(h).rolling(16, min_periods=4).mean()
    if target != "ubal_mwh":
        X["volum_t"] = a["ubal_mwh"].shift(h)
    if target != "spread":
        X["spread_t"] = k.winsor(a["spread"], 0.005).shift(h)
    if h <= SPOT_KNOWN_MAX_H:
        X["spot_tau"] = a["spot"]
        X["spot_endring"] = a["spot"] - a["spot"].shift(1)
        # hele leveringsdøgnet klareres i samme auksjon, så døgnsnittet for τ er kjent når spot(τ) er det
        X["spot_vs_dognsnitt"] = a["spot"] - a.groupby("dato")["spot"].transform("mean")
    tod = a["mtu_i_dognet"].astype(float)
    X["tod_sin"] = np.sin(2 * np.pi * tod / 96)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 96)
    X["helg"] = a["helg"].astype(float)
    X["kvarter"] = a["kvarter"].astype(float)
    return X


def climatology(a: pd.DataFrame, target: str, days: int = 28) -> pd.Series:
    """Snitt av samme kvarter-i-døgnet de siste `days` dagene (kun fortid; gyldig for h ≤ 96)."""
    y = a[target].astype(float)
    lagged = pd.concat([y.shift(96 * d) for d in range(1, days + 1)], axis=1)
    return lagged.mean(axis=1)


def _monthly_splits(index: pd.DatetimeIndex, test_start: str):
    months = pd.date_range(pd.Timestamp(test_start, tz="UTC"), index.max(), freq="MS", tz="UTC")
    for m in months:
        yield m, m + pd.offsets.MonthBegin(1)


def run_regression(a: pd.DataFrame, target: str, h: int, test_start=TEST_START) -> pd.DataFrame:
    """Returnerer en tabell med fasit og prognoser fra alle modeller for testperioden."""
    y = a[target].astype(float)
    X = features(a, target, h)
    preds = pd.DataFrame(index=a.index)
    preds["fasit"] = y
    preds["persistens"] = y.shift(h)
    preds["sesongnaiv"] = y.shift(96) if h <= 96 else np.nan
    preds["klimatologi"] = climatology(a, target)
    preds["nullmodell"] = np.nan
    preds["ARX (ridge)"] = np.nan
    preds["GBM"] = np.nan

    # robust trening: vinsoriser målet for å unngå at noen få pristopper styrer tilpasningen
    y_train_all = k.winsor(y, 0.005) if target == "spread" else y
    for m0, m1 in _monthly_splits(a.index, test_start):
        tr = (a.index >= pd.Timestamp(TRAIN_START, tz="UTC")) & (a.index < m0 - pd.Timedelta(minutes=15 * h))
        te = (a.index >= m0) & (a.index < m1)
        ok = tr & X.notna().all(axis=1) & y_train_all.notna()
        if ok.sum() < 1000 or te.sum() == 0:
            continue
        Xte = X[te].fillna(X[ok].median())
        preds.loc[te, "nullmodell"] = 0.0 if target == "spread" else float(y_train_all[ok].mean())
        lin = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[ok], y_train_all[ok])
        preds.loc[te, "ARX (ridge)"] = lin.predict(Xte)
        gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                            loss="absolute_error" if target == "spread" else "squared_error",
                                            random_state=0).fit(X[ok], y_train_all[ok])
        preds.loc[te, "GBM"] = gbm.predict(Xte)
    test = preds[preds.index >= pd.Timestamp(test_start, tz="UTC")]
    return test.dropna(subset=["fasit"])


def run_direction(a: pd.DataFrame, h: int, test_start=TEST_START) -> pd.DataFrame:
    """Klassifisering av reguleringsretning (opp = 1, ellers 0). Prognose = P(opp)."""
    r = a["retning"].astype(float)
    y = (r == 1).astype(float).where(r.notna())
    X = features(a, "retning", h)
    preds = pd.DataFrame(index=a.index)
    preds["fasit"] = y
    preds["persistens"] = y.shift(h)
    preds["sesongnaiv"] = y.shift(96)
    preds["klimatologi"] = climatology(pd.DataFrame({"y": y}), "y")
    for c in ["nullmodell", "Logistisk", "GBM"]:
        preds[c] = np.nan
    for m0, m1 in _monthly_splits(a.index, test_start):
        tr = (a.index >= pd.Timestamp(TRAIN_START, tz="UTC")) & (a.index < m0 - pd.Timedelta(minutes=15 * h))
        te = (a.index >= m0) & (a.index < m1)
        ok = tr & X.notna().all(axis=1) & y.notna()
        if ok.sum() < 1000 or te.sum() == 0:
            continue
        Xte = X[te].fillna(X[ok].median())
        preds.loc[te, "nullmodell"] = float(y[ok].mean() >= 0.5)
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(X[ok], y[ok])
        preds.loc[te, "Logistisk"] = lr.predict_proba(Xte)[:, 1]
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(X[ok], y[ok])
        preds.loc[te, "GBM"] = gb.predict_proba(Xte)[:, 1]
    test = preds[preds.index >= pd.Timestamp(test_start, tz="UTC")]
    return test.dropna(subset=["fasit"])


def score_regression(p: pd.DataFrame) -> pd.DataFrame:
    models = [c for c in p.columns if c != "fasit"]
    rows = []
    for m in models:
        d = p[["fasit", m]].dropna()
        e = d[m] - d["fasit"]
        rows.append({"modell": m, "MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()), "n": len(d)})
    s = pd.DataFrame(rows).set_index("modell")
    s["skill_vs_persistens"] = 1 - s["MAE"] / s.loc["persistens", "MAE"]
    return s


def score_direction(p: pd.DataFrame) -> pd.DataFrame:
    models = [c for c in p.columns if c != "fasit"]
    rows = []
    for m in models:
        d = p[["fasit", m]].dropna()
        rows.append({"modell": m, "treffsikkerhet": ((d[m] >= 0.5) == (d["fasit"] == 1)).mean(),
                     "Brier": ((d[m] - d["fasit"]) ** 2).mean(), "n": len(d)})
    return pd.DataFrame(rows).set_index("modell")


def run_all(areas=AREAS, horizons=HORIZONS, save: bool = True, verbose: bool = True) -> dict:
    df = k.prepare(lagring.load())
    out = {"oppsett": {"trening_fra": TRAIN_START, "test_fra": TEST_START,
                       "test_til": str(df["timestamp_utc"].max()), "horisonter_mtu": horizons},
           "resultater": []}
    for area in areas:
        a = k.area_series(df, area)
        for h in horizons:
            for target in ["spread", "ubal_mwh"]:
                s = score_regression(run_regression(a, target, h))
                for m, r in s.iterrows():
                    out["resultater"].append({"omraade": area, "h": h, "mål": target, "modell": m,
                                              **{c: float(v) for c, v in r.items()}})
            s = score_direction(run_direction(a, h))
            for m, r in s.iterrows():
                out["resultater"].append({"omraade": area, "h": h, "mål": "retning", "modell": m,
                                          **{c: float(v) for c, v in r.items()}})
            if verbose:
                print(f"  {area} h={h} ferdig")
    if save:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "benchmark.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
        pd.DataFrame(out["resultater"]).to_csv(RESULTS / "benchmark.csv", index=False)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Kjør benchmark-modellene med rullerende evaluering.")
    p.add_argument("--omraader", nargs="+", default=AREAS)
    p.add_argument("--horisonter", nargs="+", type=int, default=HORIZONS)
    args = p.parse_args(argv)
    out = run_all(args.omraader, args.horisonter)
    res = pd.DataFrame(out["resultater"])
    for target, metric in [("spread", "MAE"), ("ubal_mwh", "MAE"), ("retning", "treffsikkerhet")]:
        t = res[res["mål"] == target].pivot_table(index=["omraade", "modell"], columns="h", values=metric)
        print(f"\n=== {target}: {metric} per horisont (kvarter) ===")
        print(t.round(3).to_string())


if __name__ == "__main__":
    main()
