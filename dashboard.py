"""Dashboard for datagrunnlaget, stiliserte fakta og benchmark-resultater.

Start:  streamlit run dashboard.py
Knappen "Oppdater data nå" i sidepanelet henter nye data fra eSett.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kortmarkeder import AREAS, klargjoring as k, lagring

st.set_page_config(page_title="Korte kraftmarkeder – NO1–NO5", layout="wide")

# Fargeroller (validert kategorisk palett, fast rekkefølge per område)
AREA_COLORS = {"NO1": "#2a78d6", "NO2": "#eb6834", "NO3": "#1baf7a", "NO4": "#eda100", "NO5": "#e87ba4"}
MODEL_COLORS = {"persistens": "#898781", "nullmodell": "#c3c2b7", "sesongnaiv": "#e87ba4", "klimatologi": "#eda100",
                "ARX (ridge)": "#1baf7a", "Logistisk": "#1baf7a", "GBM": "#2a78d6"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
VAR_LABELS = {"spot": "Spotpris (EUR/MWh)", "ubal_pris": "Ubalansepris (EUR/MWh)",
              "spread": "Ubalansepris − spot (EUR/MWh)", "ubal_mwh": "Ubalansevolum (MWh/kvarter)",
              "opp_pris": "Oppreguleringspris (EUR/MWh)", "ned_pris": "Nedreguleringspris (EUR/MWh)"}


def style(fig: go.Figure, height=380, ytitle=None, xtitle=None) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="#fcfcfb",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(family="system-ui, sans-serif", color=INK, size=13),
                      legend=dict(orientation="h", y=1.08, x=0), hovermode="x unified")
    fig.update_xaxes(showgrid=False, linecolor="#c3c2b7", title=xtitle, color=MUTED)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title=ytitle, color=MUTED)
    return fig


@st.cache_data(show_spinner="Leser data …")
def load_data(mtime: float) -> pd.DataFrame:
    return k.prepare(lagring.load())


@st.cache_data
def load_json(name: str, mtime: float):
    p = lagring.ROOT / "results" / name
    return json.loads(p.read_text()) if p.exists() else None


def mtime(p):
    return p.stat().st_mtime if p.exists() else 0.0


# ---------------- sidepanel ----------------
st.sidebar.title("Korte kraftmarkeder")
st.sidebar.caption("Ubalanse- og reguleringspriser fra eSett Open Data, 15-min oppløsning")
if st.sidebar.button("Oppdater data nå", width="stretch"):
    with st.spinner("Henter fra eSett …"):
        lagring.update(verbose=False)
    st.cache_data.clear()

if not lagring.PARQUET.exists():
    st.error("Fant ingen data. Kjør `python -m kortmarkeder.lagring` først.")
    st.stop()

df = load_data(mtime(lagring.PARQUET))
last = df["timestamp_utc"].max().tz_convert(k.TZ)
st.sidebar.markdown(f"**Siste datapunkt:** {last:%d.%m.%Y kl. %H:%M}")
areas = st.sidebar.multiselect("Budområder", AREAS, default=["NO1", "NO2"])
if not areas:
    st.stop()

tab_data, tab_fakta, tab_bench, tab_status = st.tabs(
    ["Tidsserier", "Stiliserte fakta", "Benchmark-modeller", "Datastatus"])

# ---------------- tidsserier ----------------
with tab_data:
    c1, c2, c3 = st.columns([2, 1.2, 0.8])
    var = c1.selectbox("Variabel", list(VAR_LABELS), format_func=VAR_LABELS.get, index=2)
    days = c2.select_slider("Periode", options=[1, 3, 7, 14, 30, 90, 365], value=7,
                            format_func=lambda d: f"siste {d} døgn")
    zoom = c3.toggle("Skjul ekstreme topper", value=True,
                     help="Skalerer y-aksen til 0,5–99,5-persentilen. Toppene er fortsatt med i dataene.")
    start = df["timestamp_utc"].max() - pd.Timedelta(days=days)
    d = df[(df["timestamp_utc"] > start) & df["area"].isin(areas)]
    if days > 30:  # aggreger for å holde grafen lett
        d = (d.set_index("tid_lokal").groupby("area")[var].resample("1h" if days <= 90 else "1D").mean()
             .reset_index())
    fig = go.Figure()
    for a in areas:
        g = d[d["area"] == a]
        fig.add_scatter(x=g["tid_lokal"], y=g[var], name=a, mode="lines",
                        line=dict(width=2, color=AREA_COLORS[a], shape="hv" if days <= 7 else "linear"))
    fig = style(fig, 420, VAR_LABELS[var])
    if zoom and d[var].notna().any():
        lo, hi = d[var].quantile([0.005, 0.995])
        pad = 0.1 * (hi - lo)
        fig.update_yaxes(range=[lo - pad, hi + pad])
    st.plotly_chart(fig, width="stretch")

    st.subheader("Reguleringsretning per kvarter")
    d2 = df[(df["timestamp_utc"] > df["timestamp_utc"].max() - pd.Timedelta(days=min(days, 7)))
            & df["area"].isin(areas)]
    z = d2.pivot(index="area", columns="tid_lokal", values="retning").astype(float)
    fig = go.Figure(go.Heatmap(z=z.values, x=z.columns, y=z.index, zmin=-1, zmax=1,
                               colorscale=[[0, "#2a78d6"], [0.5, "#f0efec"], [1, "#e34948"]],
                               colorbar=dict(tickvals=[-1, 0, 1], ticktext=["ned", "ingen", "opp"])))
    st.plotly_chart(style(fig, 60 + 45 * len(areas)), width="stretch")

    st.subheader("Fordeling av spread (ubalansepris − spot)")
    fig = go.Figure()
    for a in areas:
        s = df[(df["area"] == a) & (df["timestamp_utc"] >= k.R3_START)]["spread"].clip(-60, 60)
        fig.add_histogram(x=s, name=a, nbinsx=120, opacity=0.6, marker_color=AREA_COLORS[a])
    fig.update_layout(barmode="overlay")
    st.plotly_chart(style(fig, 320, "antall kvarter", "EUR/MWh (klippet ved ±60)"), width="stretch")

# ---------------- stiliserte fakta ----------------
with tab_fakta:
    an = load_json("analyse.json", mtime(lagring.ROOT / "results" / "analyse.json"))
    if an is None:
        st.info("Kjør `python -m kortmarkeder.analyse` for å beregne stiliserte fakta.")
    else:
        st.caption(f"Periode: {an['periode']['fra'][:10]} – {an['periode']['til'][:10]} (15-min spot-regimet)")
        st.subheader("1 · Er prisene Brownsk bevegelse?  Variansratio VR(q)")
        st.markdown("For Brownsk bevegelse er **VR(q) = 1** for alle horisonter. VR under 1 betyr at prisen "
                    "trekkes tilbake mot et nivå (mean reversion); over 1 betyr trend.")
        col = st.radio("Serie", ["spot", "ubal_pris", "ubal_mwh"], horizontal=True,
                       format_func=VAR_LABELS.get)
        fig = go.Figure()
        for a in areas:
            vr = an["brownsk"][a][col]["variansratio"]
            fig.add_scatter(x=[int(q) / 4 for q in vr], y=list(vr.values()), name=a, mode="lines+markers",
                            line=dict(width=2, color=AREA_COLORS[a]), marker=dict(size=8))
        fig.add_hline(y=1, line=dict(color=MUTED, dash="dash"), annotation_text="Brownsk bevegelse")
        fig.update_xaxes(type="log", tickvals=[0.25, 1, 4, 24, 168, 720],
                         ticktext=["15 min", "1 t", "4 t", "1 døgn", "1 uke", "1 mnd"])
        st.plotly_chart(style(fig, 380, "VR(q)", "horisont q"), width="stretch")
        cols = st.columns(len(areas))
        for c, a in zip(cols, areas):
            b = an["brownsk"][a]
            c.metric(f"{a}: kurtose i økninger (spot)", f"{b['spot']['kurtose_okninger']:.0f}",
                     help="Normalfordeling har 0. Høy verdi = tunge haler / hopp.")
            c.metric(f"{a}: halveringstid spot-avvik", f"{b['spot']['halveringstid_dogn']:.1f} døgn")

        st.subheader("2 · Autokorrelasjon (rangbasert)")
        what = st.radio("Serie ", ["spread", "ubal_mwh", "retning", "spot_okning"], horizontal=True,
                        format_func=lambda s: {"spread": "Spread", "ubal_mwh": "Ubalansevolum",
                                               "retning": "Reguleringsretning",
                                               "spot_okning": "Endring i spot"}[s])
        fig = go.Figure()
        for a in areas:
            v = an["acf"][a][what]
            fig.add_scatter(x=np.arange(len(v)) / 4, y=v, name=a, mode="lines",
                            line=dict(width=2, color=AREA_COLORS[a]))
        st.plotly_chart(style(fig, 340, "autokorrelasjon", "lag (timer)"), width="stretch")

        st.subheader("3 · Kalender: profil gjennom døgnet (snitt)")
        cv = st.radio("Størrelse", ["spot", "spread_w", "ubal_mwh", "opp"], horizontal=True,
                      format_func=lambda s: {"spot": "Spot", "spread_w": "Spread", "ubal_mwh": "Ubalansevolum",
                                             "opp": "Andel oppregulering"}[s])
        fig = go.Figure()
        x = np.arange(96) / 4
        for a in areas:
            for day, dash in [("hverdag", "solid"), ("helg", "dot")]:
                fig.add_scatter(x=x, y=an["kalender"][a][day][cv], name=f"{a} {day}", mode="lines",
                                line=dict(width=2, color=AREA_COLORS[a], dash=dash))
        st.plotly_chart(style(fig, 340, None, "time i døgnet (lokal tid)"), width="stretch")

        st.subheader("4 · Pris mot volum: spread som funksjon av ubalanse")
        a0 = areas[0]
        pv = an["pris_volum"][a0]
        fig = go.Figure()
        fig.add_bar(x=pv["bins"], y=pv["median"], name="median spread", marker_color=AREA_COLORS[a0])
        fig.add_scatter(x=pv["bins"], y=pv["p10"], mode="markers", name="10-persentil",
                        marker=dict(color=MUTED, size=8, symbol="line-ew-open"))
        fig.add_scatter(x=pv["bins"], y=pv["p90"], mode="markers", name="90-persentil",
                        marker=dict(color=INK, size=8, symbol="line-ew-open"))
        st.plotly_chart(style(fig, 340, "EUR/MWh", f"ubalansevolum {a0} (MWh/kvarter, + = overskudd)"),
                        width="stretch")

        st.subheader("5 · Hendelser på tvers av områder")
        same = pd.DataFrame(an["omraader"]["samme_retning"]).loc[AREAS, AREAS]
        fig = go.Figure(go.Heatmap(z=same.values, x=AREAS, y=AREAS, zmin=0.5, zmax=1,
                                   colorscale=[[0, "#cde2fb"], [1, "#104281"]],
                                   text=same.round(2).values, texttemplate="%{text}"))
        st.plotly_chart(style(fig, 320), width="stretch")
        st.caption("Andel kvarter der to områder har samme reguleringsretning.")
        cl = pd.DataFrame(an["omraader"]["topp_klynging"]).T
        st.dataframe(cl.rename(columns={"p_topp_neste_time_gitt_topp": "P(topp neste time | topp nå)",
                                        "p_topp_neste_time_ubetinget": "P(topp neste time)",
                                        "terskel_eur": "terskel (EUR/MWh, 99,5-pst.)",
                                        "antall_topper": "antall topper"}).round(3))

# ---------------- benchmark ----------------
with tab_bench:
    bm = load_json("benchmark.json", mtime(lagring.ROOT / "results" / "benchmark.json"))
    if bm is None:
        st.info("Kjør `python -m kortmarkeder.benchmark` for å evaluere modellene.")
    else:
        res = pd.DataFrame(bm["resultater"])
        st.caption(f"Trening fra {bm['oppsett']['trening_fra']}, test fra {bm['oppsett']['test_fra']}, "
                   "månedlig re-tilpasning på utvidende vindu.")
        target = st.radio("Mål", ["spread", "ubal_mwh", "retning"], horizontal=True,
                          format_func=lambda s: {"spread": "Ubalansepris − spot", "ubal_mwh": "Ubalansevolum",
                                                 "retning": "Reguleringsretning"}[s])
        a0 = st.selectbox("Område", [a for a in areas if a in res["omraade"].unique()] or AREAS)
        metric = "treffsikkerhet" if target == "retning" else "MAE"
        r = res[(res["mål"] == target) & (res["omraade"] == a0)]
        fig = go.Figure()
        for m, g in r.groupby("modell"):
            g = g.sort_values("h")
            fig.add_scatter(x=g["h"] / 4, y=g[metric], name=m, mode="lines+markers",
                            line=dict(width=2, color=MODEL_COLORS.get(m, MUTED)), marker=dict(size=8))
        fig.update_xaxes(type="log", tickvals=[0.25, 1, 4, 24], ticktext=["15 min", "1 t", "4 t", "1 døgn"])
        st.plotly_chart(style(fig, 380, metric + (" (høyere er bedre)" if target == "retning"
                                                  else " (lavere er bedre)"), "prognosehorisont"),
                        width="stretch")
        st.dataframe(r.pivot_table(index="modell", columns="h", values=metric).round(3)
                     .rename(columns=lambda h: f"h={h} ({h * 15} min)"))

# ---------------- datastatus ----------------
with tab_status:
    st.subheader("Dekning per område")
    cov = df.groupby("area").agg(fra=("tid_lokal", "min"), til=("tid_lokal", "max"), rader=("spot", "size"),
                                 mangler_volum=("ubal_mwh", lambda s: int(s.isna().sum())))
    st.dataframe(cov)
    st.subheader("Markedsdesign-regimer i dataene")
    st.markdown("""
- **R1 (før 4. mars 2025):** ubalansepris i praksis per time.
- **R2 (4. mars – 30. sept. 2025):** nordisk mFRR-energiaktiveringsmarked – ubalansepris per kvarter, spot per time.
- **R3 (fra 1. okt. 2025):** 15-minutters day-ahead – hele kjeden i kvartersoppløsning.
""")
    an = load_json("analyse.json", mtime(lagring.ROOT / "results" / "analyse.json"))
    if an:
        m = an["maanedlig"]
        fig = go.Figure()
        for a in areas:
            fig.add_scatter(x=m[a]["mnd"], y=m[a]["andel_ubal_endres"], name=a, mode="lines",
                            line=dict(width=2, color=AREA_COLORS[a]))
        st.plotly_chart(style(fig, 300, "andel kvarter der ubalanseprisen endres"), width="stretch")
