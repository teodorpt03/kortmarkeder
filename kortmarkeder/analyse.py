"""Stiliserte fakta for de korte markedene - svar på spørsmålene vi startet med.

1. Oppfører prisene seg som Brownsk bevegelse (random walk)?
   - Variansratio VR(q) = Var(X[t+q] − X[t]) / (q · Var(ΔX)). For BM er VR(q) = 1 for alle q.
     VR < 1 betyr at prisen trekkes tilbake (mean reversion), VR > 1 betyr trend/momentum.
   - Kurtose i økningene (BM: normalfordelt, kurtose 0) og ADF-test for enhetsrot.
   - Halveringstid for avvik (AR(1) / Ornstein–Uhlenbeck-tolkning).
   NB: Kraftpriser kan bli negative, så log-avkastning (geometrisk BM) gir ikke mening.

2. Er prisene forutsigbare? Autokorrelasjon (rangbasert, robust mot pristopper) for spread,
   volum og retning, og retning som Markov-kjede.

3. Kalendertid: profil over døgnet, helg/ukedag, og kvarter-innen-timen før/etter 15-min spot.

4. Volum: hvor forutsigbart er ubalansevolumet, og hvordan henger det sammen med prisen?

5. Hendelser mot hverandre: samvariasjon mellom budområder, pristopper som klynger seg.

Kjør:  python -m kortmarkeder.analyse   → results/analyse.json
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf, adfuller

from . import AREAS, klargjoring as k, lagring

warnings.filterwarnings("ignore")
RESULTS = lagring.ROOT / "results"
VR_LAGS = [1, 2, 4, 8, 16, 32, 48, 96, 192, 384, 672, 1344, 2880]
ACF_LAGS = 192


def variance_ratio(x: np.ndarray, q: int) -> float:
    d1 = np.diff(x)
    dq = x[q:] - x[:-q]
    return float(np.nanvar(dq) / (q * np.nanvar(d1)))


def half_life(x: pd.Series) -> float:
    m = x.notna() & x.shift(1).notna()
    phi = np.corrcoef(x[m], x.shift(1)[m])[0, 1]
    return float(-np.log(2) / np.log(phi)) if 0 < phi < 1 else float("nan")


def rank_acf(s: pd.Series, nlags: int = ACF_LAGS) -> list[float]:
    """Spearman-type autokorrelasjon: ACF av rangerte verdier (robust mot ekstreme topper)."""
    r = s.rank().interpolate(limit_direction="both")
    return [float(v) for v in acf(r.values, nlags=nlags, fft=True)]


def brownian_tests(a: pd.DataFrame) -> dict:
    out = {}
    for col in ["spot", "ubal_pris", "ubal_mwh"]:
        x = a[col].interpolate(limit_direction="both")
        dx = np.diff(x.values)
        adf = adfuller(x.values, maxlag=96, autolag=None)
        out[col] = {
            "variansratio": {str(q): variance_ratio(x.values, q) for q in VR_LAGS},
            "kurtose_okninger": float(stats.kurtosis(dx)),
            "andel_okninger_lik_null": float(np.mean(dx == 0)),
            "adf_p": float(adf[1]),
            "acf_okninger_lag1": float(acf(dx, nlags=1, fft=True)[1]),
        }
    # halveringstid for spot: avvik fra 30-dagers glidende snitt, på døgnnivå
    dm = a["spot"].resample("D").mean()
    out["spot"]["halveringstid_dogn"] = half_life(dm - dm.rolling(30, min_periods=10).mean())
    out["spread_halveringstid_kvarter"] = half_life(k.winsor(a["spread"], 0.005))
    return out


def qq_points(a: pd.DataFrame, col: str = "spot", n: int = 60) -> dict:
    dx = a[col].interpolate().diff().dropna()
    z = (dx - dx.mean()) / dx.std()
    probs = np.linspace(0.005, 0.995, n)
    return {"teoretisk": stats.norm.ppf(probs).round(3).tolist(),
            "empirisk": np.quantile(z, probs).round(3).tolist()}


def markov(a: pd.DataFrame) -> dict:
    r = a["retning"].dropna().astype(int)
    r = r[r != 0]
    tm = pd.crosstab(r.shift(1), r, normalize="index")
    runs = r.groupby((r != r.shift()).cumsum()).agg(["first", "size"])
    hist = runs["size"].clip(upper=40).value_counts().sort_index()
    return {"overgang": {str(int(i)): {str(int(j)): float(tm.loc[i, j]) for j in tm.columns} for i in tm.index},
            "snitt_lengde_kvarter": float(runs["size"].mean()),
            "median_lengde_kvarter": float(runs["size"].median()),
            "lengste_kvarter": int(runs["size"].max()),
            "lengde_hist": {str(int(i)): int(v) for i, v in hist.items()}}


def calendar(df_area: pd.DataFrame) -> dict:
    d = df_area.copy()
    d["spread_w"] = k.winsor(d["spread"], 0.005)
    d["opp"] = (d["retning"] == 1).astype(float)
    prof = {}
    for helg, g in d.groupby("helg"):
        p = g.groupby("mtu_i_dognet")[["spot", "spread_w", "ubal_mwh", "opp"]].mean()
        prof["helg" if helg else "hverdag"] = {c: p[c].round(3).tolist() for c in p.columns}
    return prof


def quarter_effect(df: pd.DataFrame, area: str) -> dict:
    out = {}
    for name, lo, hi in [("R2 (timesspot)", "2025-03-04", "2025-10-01"),
                         ("R3 (15-min spot)", "2025-10-01", "2100-01-01")]:
        g = df[(df["area"] == area) & (df["timestamp_utc"] >= pd.Timestamp(lo, tz="UTC"))
               & (df["timestamp_utc"] < pd.Timestamp(hi, tz="UTC"))].copy()
        g["abs_spread"] = k.winsor(g["spread"], 0.005).abs()
        g["abs_vol"] = g["ubal_mwh"].abs()
        q = g.groupby("kvarter")[["abs_spread", "abs_vol"]].mean()
        out[name] = {c: q[c].round(3).tolist() for c in q.columns}
    return out


def price_volume(df_area: pd.DataFrame) -> dict:
    d = df_area.dropna(subset=["ubal_mwh"]).copy()
    edges = [-1e9, -60, -40, -20, -10, 0, 10, 20, 40, 60, 1e9]
    labels = ["< −60", "−60–−40", "−40–−20", "−20–−10", "−10–0", "0–10", "10–20", "20–40", "40–60", "> 60"]
    d["bin"] = pd.cut(d["ubal_mwh"], edges, labels=labels)
    g = d.groupby("bin", observed=False).agg(n=("spread", "size"), median=("spread", "median"),
                                             p10=("spread", lambda s: s.quantile(.1)),
                                             p90=("spread", lambda s: s.quantile(.9)),
                                             p_opp=("retning", lambda s: (s == 1).mean()))
    return {"bins": labels, **{c: g[c].round(3).tolist() for c in g.columns}}


def cross_area(df: pd.DataFrame) -> dict:
    R = k.wide(df, "retning").astype(float)
    same = {a: {b: float((R[a] == R[b]).mean()) for b in R.columns} for a in R.columns}
    S = k.wide(df, "spread")
    corr = S.rank().corr().round(3)
    spikes = S.gt(S.quantile(0.995))
    co = {a: {b: float(spikes.loc[spikes[a], b].mean()) for b in S.columns} for a in S.columns}
    clustering = {}
    for a in S.columns:
        sp = spikes[a]
        nxt = sp.rolling(4).max().shift(-4).fillna(0).astype(bool)
        clustering[a] = {"p_topp_neste_time_gitt_topp": float(nxt[sp].mean()),
                         "p_topp_neste_time_ubetinget": float(nxt.mean()),
                         "terskel_eur": float(S[a].quantile(0.995)),
                         "antall_topper": int(sp.sum())}
    return {"samme_retning": same, "spread_rangkorrelasjon": corr.to_dict(),
            "topp_samtidig": co, "topp_klynging": clustering}


def monthly_regimes(df: pd.DataFrame) -> dict:
    d = df.copy()
    d["mnd"] = d["tid_lokal"].dt.strftime("%Y-%m")
    d["abs_spread"] = d["spread"].abs()
    d["dspot_nonzero"] = d.groupby("area")["spot"].diff().ne(0)
    d["dimbl_nonzero"] = d.groupby("area")["ubal_pris"].diff().ne(0)
    g = d.groupby(["area", "mnd"]).agg(spot=("spot", "mean"), median_abs_spread=("abs_spread", "median"),
                                       andel_spot_endres=("dspot_nonzero", "mean"),
                                       andel_ubal_endres=("dimbl_nonzero", "mean"))
    out = {}
    for area in d["area"].unique():
        x = g.loc[area]
        out[area] = {"mnd": x.index.tolist(), **{c: x[c].round(3).tolist() for c in x.columns}}
    return out


def same_season_comparison(df: pd.DataFrame) -> dict:
    """Samme sesong før/etter 15-min spot (mars–sept. 2025 mot mars–sept. 2026), for å unngå sesongforveksling."""
    out = {}
    for name, lo, hi in [("mar–sep 2025 (timesspot)", "2025-03-04", "2025-10-01"),
                         ("mar–sep 2026 (15-min spot)", "2026-03-04", "2026-10-01")]:
        d = df[(df["timestamp_utc"] >= pd.Timestamp(lo, tz="UTC")) & (df["timestamp_utc"] < pd.Timestamp(hi, tz="UTC"))]
        g = d.groupby("area").agg(snitt_spot=("spot", "mean"),
                                  median_abs_spread=("spread", lambda s: s.abs().median()),
                                  andel_spread_over_100=("spread", lambda s: (s > 100).mean()),
                                  snitt_abs_ubal_mwh=("ubal_mwh", lambda s: s.abs().mean()))
        out[name] = {a: {c: float(g.loc[a, c]) for c in g.columns} for a in g.index}
    return out


def summary_table(df: pd.DataFrame) -> dict:
    out = {}
    for area, g in df.groupby("area"):
        out[area] = {"snitt_spot": float(g["spot"].mean()), "snitt_ubal_pris": float(g["ubal_pris"].mean()),
                     "median_abs_spread": float(g["spread"].abs().median()),
                     "maks_ubal_pris": float(g["ubal_pris"].max()), "min_ubal_pris": float(g["ubal_pris"].min()),
                     "andel_opp": float((g["retning"] == 1).mean()), "andel_ned": float((g["retning"] == -1).mean()),
                     "andel_negativ_spot": float((g["spot"] < 0).mean()),
                     "snitt_abs_ubal_mwh": float(g["ubal_mwh"].abs().mean())}
    return out


def run(save: bool = True) -> dict:
    df = k.prepare(lagring.load())
    r3 = df[df["timestamp_utc"] >= k.R3_START]
    out = {"periode": {"fra": str(r3["timestamp_utc"].min()), "til": str(r3["timestamp_utc"].max()),
                       "kommentar": "Hovedanalysen bruker R3 (fra 15-min spot 1.10.2025)"},
           "oppsummering": summary_table(r3), "brownsk": {}, "acf": {}, "markov": {},
           "kalender": {}, "kvarter": {}, "pris_volum": {}}
    for area in AREAS:
        a = k.area_series(r3, area)
        out["brownsk"][area] = brownian_tests(a)
        out["acf"][area] = {"spread": rank_acf(a["spread"]), "ubal_mwh": rank_acf(a["ubal_mwh"]),
                            "retning": [float(v) for v in acf(a["retning"].astype(float).interpolate(
                                limit_direction="both").values, nlags=ACF_LAGS, fft=True)],
                            "spot_okning": [float(v) for v in acf(a["spot"].interpolate().diff().dropna().values,
                                                                  nlags=ACF_LAGS, fft=True)]}
        out["markov"][area] = markov(a)
        out["kalender"][area] = calendar(r3[r3["area"] == area])
        out["kvarter"][area] = quarter_effect(df, area)
        out["pris_volum"][area] = price_volume(r3[r3["area"] == area])
    out["qq_spot_NO1"] = qq_points(k.area_series(r3, "NO1"))
    out["omraader"] = cross_area(r3)
    out["maanedlig"] = monthly_regimes(df)
    out["sesong_sammenligning"] = same_season_comparison(df)
    if save:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "analyse.json").write_text(json.dumps(out, ensure_ascii=False))
    return out


def main():
    out = run()
    print("Variansratio for spot (BM ⇒ 1):")
    for area in AREAS:
        vr = out["brownsk"][area]["spot"]["variansratio"]
        print(f"  {area}: " + "  ".join(f"q={q}:{vr[str(q)]:.2f}" for q in [4, 16, 96, 672, 2880]))
    print("Markov – P(samme retning neste kvarter):",
          {a: round(out["markov"][a]["overgang"]["1"]["1"], 3) for a in AREAS})
    print(f"Lagret i {RESULTS / 'analyse.json'}")


if __name__ == "__main__":
    main()
