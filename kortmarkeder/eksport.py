"""Eksporter resultatene til flate CSV-tabeller (for rapportsiden og videre bruk i f.eks. Excel).

Kjør:  python -m kortmarkeder.eksport   → results/tabeller/*.csv
"""
from __future__ import annotations

import json

import pandas as pd

from . import AREAS, lagring

RESULTS = lagring.ROOT / "results"
OUT = RESULTS / "tabeller"


def main():
    an = json.loads((RESULTS / "analyse.json").read_text())
    bm = json.loads((RESULTS / "benchmark.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    t = {}

    rows = []
    for a in AREAS:
        s, b, m = an["oppsummering"][a], an["brownsk"][a], an["markov"][a]
        rows.append({"omraade": a, "snitt_spot": s["snitt_spot"], "snitt_ubal_pris": s["snitt_ubal_pris"],
                     "median_abs_spread": s["median_abs_spread"], "maks_ubal_pris": s["maks_ubal_pris"],
                     "min_ubal_pris": s["min_ubal_pris"], "andel_opp": s["andel_opp"], "andel_ned": s["andel_ned"],
                     "snitt_abs_ubal_mwh": s["snitt_abs_ubal_mwh"],
                     "kurtose_spot_okninger": b["spot"]["kurtose_okninger"],
                     "kurtose_ubal_pris_okninger": b["ubal_pris"]["kurtose_okninger"],
                     "adf_p_spot": b["spot"]["adf_p"],
                     "halveringstid_spot_dogn": b["spot"]["halveringstid_dogn"],
                     "halveringstid_spread_min": b["spread_halveringstid_kvarter"] * 15,
                     "p_samme_retning": (m["overgang"]["1"]["1"] + m["overgang"]["-1"]["-1"]) / 2,
                     "snitt_retningsperiode_min": m["snitt_lengde_kvarter"] * 15,
                     "median_retningsperiode_min": m["median_lengde_kvarter"] * 15})
    t["oppsummering"] = pd.DataFrame(rows)

    t["variansratio"] = pd.DataFrame([
        {"omraade": a, "serie": serie, "q_kvarter": int(q), "horisont_timer": int(q) / 4, "vr": v}
        for a in AREAS for serie in ["spot", "ubal_pris", "ubal_mwh"]
        for q, v in an["brownsk"][a][serie]["variansratio"].items()])

    t["autokorrelasjon"] = pd.DataFrame([
        {"omraade": a, "serie": serie, "lag_kvarter": i, "lag_timer": i / 4, "acf": v}
        for a in AREAS for serie in ["spread", "ubal_mwh", "retning"]
        for i, v in enumerate(an["acf"][a][serie])])

    t["markov"] = pd.DataFrame([
        {"omraade": a, "fra": int(i), "til": int(j), "sannsynlighet": p}
        for a in AREAS for i, row in an["markov"][a]["overgang"].items() for j, p in row.items()])

    t["dognprofil"] = pd.DataFrame([
        {"omraade": a, "dagtype": dt, "mtu": i, "time": i / 4,
         "spot": p["spot"][i], "spread": p["spread_w"][i], "ubal_mwh": p["ubal_mwh"][i], "andel_opp": p["opp"][i]}
        for a in AREAS for dt, p in an["kalender"][a].items() for i in range(96)])

    t["prisvolum"] = pd.DataFrame([
        {"omraade": a, "rekkefolge": i, "volumintervall": pv["bins"][i], "antall": pv["n"][i],
         "median": pv["median"][i], "p10": pv["p10"][i], "p90": pv["p90"][i], "andel_opp": pv["p_opp"][i]}
        for a in AREAS for pv in [an["pris_volum"][a]] for i in range(len(pv["bins"]))])

    t["samme_retning"] = pd.DataFrame([
        {"omraade_a": a, "omraade_b": b, "andel": v}
        for a, row in an["omraader"]["samme_retning"].items() for b, v in row.items()])

    t["topper"] = pd.DataFrame([{"omraade": a, **v} for a, v in an["omraader"]["topp_klynging"].items()])

    t["maanedlig"] = pd.DataFrame([
        {"omraade": a, "maaned": mnd, "spot": m["spot"][i], "median_abs_spread": m["median_abs_spread"][i],
         "andel_ubal_endres": m["andel_ubal_endres"][i], "andel_spot_endres": m["andel_spot_endres"][i]}
        for a, m in an["maanedlig"].items() for i, mnd in enumerate(m["mnd"])])

    t["sesong_sammenligning"] = pd.DataFrame([
        {"periode": per, "omraade": a, **v}
        for per, d in an["sesong_sammenligning"].items() for a, v in d.items()])

    b = pd.DataFrame(bm["resultater"]).rename(columns={"mål": "maal"})
    b["horisont"] = b["h"].map({1: "15 min", 4: "1 time", 16: "4 timer", 96: "1 døgn"})
    t["benchmark"] = b

    for name, df in t.items():
        df.round(5).to_csv(OUT / f"{name}.csv", index=False)
        print(f"  {name}.csv  ({len(df)} rader)")


if __name__ == "__main__":
    main()
