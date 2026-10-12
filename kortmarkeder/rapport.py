"""Lager en frittstående rapportside (docs/index.html) fra resultatene.

Siden kan publiseres gratis med GitHub Pages (Settings → Pages → Deploy from a branch → main, /docs),
og åpnes da av alle uten innlogging. Figurene er interaktive (Plotly), og budområde velges øverst.

Kjør etter analyse og benchmark:
    python -m kortmarkeder.analyse
    python -m kortmarkeder.benchmark
    python -m kortmarkeder.eksport
    python -m kortmarkeder.rapport
"""
from __future__ import annotations

import html
import json

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from . import AREAS, lagring

T = lagring.ROOT / "results" / "tabeller"
OUT = lagring.ROOT / "docs" / "index.html"

AREA_COLORS = dict(zip(AREAS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#898781"]
MUTED, INK, GRID = "#b9b8b1", "#1d1c1a", "#ebeae4"
HOURS = {0.25: "15 min", 1: "1 t", 4: "4 t", 24: "1 døgn", 168: "1 uke", 720: "1 mnd"}
H_NAMES = {1: "15 min", 4: "1 time", 16: "4 timer", 96: "1 døgn"}


def layout(fig: go.Figure, ytitle="", xtitle="", height=330, legend=True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=INK), hovermode="x unified",
        showlegend=legend, legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(size=11)),
        hoverlabel=dict(font=dict(family="Inter, sans-serif")))
    fig.update_xaxes(title=xtitle, showgrid=False, linecolor="#c3c2b7", ticks="", color="#6b6a65", title_font_size=11)
    fig.update_yaxes(title=ytitle, gridcolor=GRID, zeroline=False, color="#6b6a65", title_font_size=11)
    return fig


def div(fig: go.Figure, fid: str) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=fid,
                       config={"displayModeBar": False, "responsive": True})


def per_area(fig: go.Figure, area: str, **kw):
    """Legg til et spor merket med område (vises/skjules av områdevelgeren)."""
    fig.add_trace(go.Scatter(meta=area, visible=(area == "NO1"), **kw))


def build() -> str:
    opp = pd.read_csv(T / "oppsummering.csv").set_index("omraade")
    vr = pd.read_csv(T / "variansratio.csv")
    ac = pd.read_csv(T / "autokorrelasjon.csv")
    dg = pd.read_csv(T / "dognprofil.csv")
    pv = pd.read_csv(T / "prisvolum.csv")
    sr = pd.read_csv(T / "samme_retning.csv")
    mn = pd.read_csv(T / "maanedlig.csv")
    bm = pd.read_csv(T / "benchmark.csv")
    figs = {}

    # 1 variansratio
    f = go.Figure()
    for a in AREAS:
        for i, (s, n) in enumerate([("spot", "Spotpris"), ("ubal_pris", "Ubalansepris"), ("ubal_mwh", "Ubalansevolum")]):
            d = vr[(vr.omraade == a) & (vr.serie == s)].sort_values("horisont_timer")
            per_area(f, a, x=d.horisont_timer, y=d.vr, name=n, mode="lines+markers", legendgroup=n,
                     showlegend=(a == "NO1"), line=dict(color=SERIES[i], width=2), marker=dict(size=6),
                     hovertemplate="%{y:.2f}<extra>" + n + "</extra>")
    f.add_hline(y=1, line=dict(color=INK, dash="dash", width=1), annotation_text="Brownsk bevegelse: VR = 1",
                annotation_position="bottom right", annotation_font_size=11)
    f.update_xaxes(type="log", tickvals=list(HOURS), ticktext=list(HOURS.values()))
    figs["vr"] = layout(f, "Variansratio VR(q)", "Horisont q")

    # 2 autokorrelasjon
    f = go.Figure()
    for a in AREAS:
        for i, (s, n) in enumerate([("spread", "Ubalansepris − spot"), ("ubal_mwh", "Ubalansevolum"),
                                    ("retning", "Reguleringsretning")]):
            d = ac[(ac.omraade == a) & (ac.serie == s)]
            per_area(f, a, x=d.lag_timer, y=d.acf, name=n, mode="lines", legendgroup=n, showlegend=(a == "NO1"),
                     line=dict(color=SERIES[i], width=2), hovertemplate="%{y:.2f}<extra>" + n + "</extra>")
    f.update_xaxes(tickvals=list(range(0, 49, 6)), ticksuffix=" t")
    figs["acf"] = layout(f, "Autokorrelasjon (rangbasert)", "Lag")

    # 3 døgnprofil: andel oppregulering
    f = go.Figure()
    for a in AREAS:
        for dt, dash, n in [("hverdag", "solid", "Hverdag"), ("helg", "dot", "Helg")]:
            d = dg[(dg.omraade == a) & (dg.dagtype == dt)]
            per_area(f, a, x=d.time, y=d.andel_opp, name=n, mode="lines", legendgroup=n, showlegend=(a == "NO1"),
                     line=dict(color=SERIES[0], width=2, dash=dash), hovertemplate="%{y:.0%}<extra>" + n + "</extra>")
    f.add_hline(y=0.5, line=dict(color="#c3c2b7", width=1))
    f.update_xaxes(tickvals=list(range(0, 25, 3)), ticktext=[f"{h:02d}:00" for h in range(0, 25, 3)])
    f.update_yaxes(tickformat=".0%")
    figs["dogn"] = layout(f, "Andel kvarter med oppregulering", "Klokkeslett (lokal tid)")

    # 4 pris mot volum
    f = go.Figure()
    for a in AREAS:
        d = pv[pv.omraade == a].sort_values("rekkefolge")
        f.add_trace(go.Bar(meta=a, visible=(a == "NO1"), x=d.volumintervall, y=d["median"], name="Median",
                           marker_color=[SERIES[1] if m >= 0 else SERIES[0] for m in d["median"]],
                           error_y=dict(type="data", symmetric=False, array=d.p90 - d["median"],
                                        arrayminus=d["median"] - d.p10, color="#6b6a65", thickness=1.2, width=4),
                           customdata=d[["p10", "p90", "andel_opp"]],
                           hovertemplate="Median %{y:.1f} EUR/MWh<br>10–90-pst.: %{customdata[0]:.1f} til "
                                         "%{customdata[1]:.1f}<br>Andel opp: %{customdata[2]:.0%}<extra></extra>"))
    figs["pv"] = layout(f, "Ubalansepris − spot, EUR/MWh", "Ubalansevolum, MWh per kvarter (+ = overskudd)",
                        legend=False)

    # 5 benchmark
    for target, ytitle, key in [("spread", "MAE, EUR/MWh", "bm_pris"), ("ubal_mwh", "MAE, MWh per kvarter", "bm_vol")]:
        f = go.Figure()
        models = ["persistens", "nullmodell", "sesongnaiv", "klimatologi", "ARX (ridge)", "GBM"]
        for a in AREAS:
            for i, m in enumerate(models):
                d = bm[(bm.omraade == a) & (bm.maal == target) & (bm.modell == m)].sort_values("h")
                col = {"GBM": SERIES[0], "nullmodell": SERIES[1], "persistens": "#4d4c48"}.get(m, MUTED)
                per_area(f, a, x=[H_NAMES[h] for h in d.h], y=d.MAE, name=m, mode="lines+markers", legendgroup=m,
                         showlegend=(a == "NO1"), line=dict(color=col, width=2.5 if col != MUTED else 1.3,
                                                            dash="dot" if col == MUTED else "solid"),
                         marker=dict(size=7 if col != MUTED else 4), hovertemplate="%{y:.2f}<extra>" + m + "</extra>")
        figs[key] = layout(f, ytitle, "Prognosehorisont")

    # 6 samme retning (statisk)
    z = sr.pivot(index="omraade_a", columns="omraade_b", values="andel").loc[AREAS, AREAS]
    f = go.Figure(go.Heatmap(z=z.values, x=AREAS, y=AREAS, zmin=0.5, zmax=1, text=(z.values * 100).round(0),
                             texttemplate="%{text:.0f} %", colorscale=[[0, "#eef4fc"], [1, "#184f95"]],
                             showscale=False, hovertemplate="%{y} og %{x}: %{z:.0%}<extra></extra>"))
    f.update_yaxes(autorange="reversed")
    figs["samme"] = layout(f, height=300, legend=False)
    figs["samme"].update_layout(hovermode="closest")

    # 7 månedlig median |spread| (alle områder)
    f = go.Figure()
    for a in AREAS:
        d = mn[mn.omraade == a]
        f.add_trace(go.Scatter(x=d.maaned, y=d.median_abs_spread, name=a, mode="lines",
                               line=dict(color=AREA_COLORS[a], width=2), hovertemplate="%{y:.1f}<extra>" + a + "</extra>"))
    for x, t in [("2025-03", "mFRR EAM"), ("2025-10", "15-min spot")]:
        f.add_vline(x=x, line=dict(color="#a8a79f", dash="dot", width=1))
        f.add_annotation(x=x, y=1, yref="paper", text=t, showarrow=False, xanchor="left", font_size=11)
    figs["mnd"] = layout(f, "Median |ubalansepris − spot|, EUR/MWh")

    # nøkkeltall per område (til JS)
    keys = {a: {"spot": f"{opp.loc[a, 'snitt_spot']:.0f}", "spread": f"{opp.loc[a, 'median_abs_spread']:.1f}",
                "markov": f"{opp.loc[a, 'p_samme_retning'] * 100:.0f} %",
                "hl": f"{opp.loc[a, 'halveringstid_spot_dogn']:.1f}".replace(".", ","),
                "skill": f"{bm[(bm.omraade == a) & (bm.maal == 'spread') & (bm.h == 4) & (bm.modell == 'GBM')].skill_vs_persistens.iloc[0] * 100:.0f} %"}
            for a in AREAS}
    gbm = bm[(bm.maal == "spread") & (bm.h == 4) & (bm.modell == "GBM")].skill_vs_persistens
    skill_rng = f"{gbm.min() * 100:.0f}–{gbm.max() * 100:.0f}"

    D = {k: div(v, k) for k, v in figs.items()}
    return TEMPLATE.format(plotly=get_plotlyjs(), keys=json.dumps(keys), skill_rng=html.escape(skill_rng), **D)


TEMPLATE = """<!doctype html>
<html lang="no"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pris og volum i korte kraftmarkeder</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Lora:wght@500&display=swap" rel="stylesheet">
<script>{plotly}</script>
<style>
:root {{ --ink:#1d1c1a; --muted:#6b6a65; --rule:#e3e1da; --blue:#2a78d6; --wash:#f6f5f1; --bg:#fdfdfb; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:Inter, system-ui, sans-serif; font-size:16px; line-height:1.55; }}
main {{ max-width:960px; margin:0 auto; padding:40px 20px 64px; }}
.eyebrow {{ font-size:12px; letter-spacing:.07em; text-transform:uppercase; color:var(--muted); }}
h1 {{ font-family:Lora, Georgia, serif; font-weight:500; font-size:clamp(30px,5vw,42px); line-height:1.1; margin:8px 0 10px; letter-spacing:-.01em; text-wrap:balance; }}
h2 {{ font-size:20px; font-weight:600; margin:0 0 6px; text-wrap:balance; }}
h3 {{ font-size:17px; font-weight:600; margin:22px 0 4px; }}
h2 .n {{ font-family:Lora, serif; color:var(--blue); margin-right:8px; }}
p {{ margin:0 0 10px; max-width:68ch; }}
.sub {{ color:var(--muted); font-size:14px; margin-bottom:18px; }}
.bar {{ position:sticky; top:0; z-index:5; background:var(--bg); border-bottom:1px solid var(--rule); padding:10px 0; margin:24px 0 8px; display:flex; flex-wrap:wrap; gap:12px; align-items:center; }}
.seg {{ display:inline-flex; border:1px solid var(--rule); border-radius:8px; overflow:hidden; }}
.seg button {{ font:inherit; font-size:14px; border:0; background:transparent; padding:6px 14px; cursor:pointer; color:var(--muted); border-right:1px solid var(--rule); }}
.seg button:last-child {{ border-right:0; }}
.seg button[aria-pressed=true] {{ background:var(--ink); color:#fff; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:18px 0 8px; }}
.kpi {{ background:var(--wash); border-radius:10px; padding:12px 14px; }}
.kpi .l {{ font-size:12.5px; color:var(--muted); line-height:1.35; }}
.kpi .v {{ font-family:Lora, serif; font-size:28px; font-variant-numeric:tabular-nums; }}
section {{ padding:28px 0; border-bottom:1px solid var(--rule); }}
.grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr)); gap:20px; }}
.cap {{ font-size:13.5px; color:var(--muted); }}
.note {{ background:var(--wash); border-radius:10px; padding:14px 16px; font-size:14.5px; margin-top:24px; }}
footer {{ font-size:14px; color:var(--muted); padding-top:24px; }}
a {{ color:var(--blue); }}
</style></head>
<body><main>
<div class="eyebrow">Miniprosjekt · Teodor Pedersen Thorsrud · søknad sommerforsker SINTEF Energi 2027</div>
<h1>Pris og volum i korte kraftmarkeder</h1>
<div class="sub">Ubalanse- og spotpriser, reguleringsretning og ubalansevolum for NO1–NO5 · eSett Open Data · 15 min · analyseperiode 1.10.2025–9.10.2026</div>
<p>En liten versjon av arbeidsflyten i SINTEF Energis prosjekt «Datadrevet prediksjon av pris og volum i korte kraftmarkeder»: automatisk datainnhenting, et dashbord og seks prognosemodeller evaluert med rullerende vindu. Kode og metode: <a href="https://github.com/teodorpt03/kortmarkeder">github.com/teodorpt03/kortmarkeder</a>.</p>

<div class="bar"><span class="cap">Budområde</span><div class="seg" id="area" role="group" aria-label="Budområde"></div></div>

<div class="kpis">
  <div class="kpi"><div class="l">Snitt spotpris, EUR/MWh</div><div class="v" data-k="spot">–</div></div>
  <div class="kpi"><div class="l">Median |ubalansepris − spot|, EUR/MWh</div><div class="v" data-k="spread">–</div></div>
  <div class="kpi"><div class="l">Samme reguleringsretning neste kvarter</div><div class="v" data-k="markov">–</div></div>
  <div class="kpi"><div class="l">Halveringstid for spotavvik, døgn</div><div class="v" data-k="hl">–</div></div>
  <div class="kpi"><div class="l">Prognose 1 time frem: lavere feil enn persistens</div><div class="v" data-k="skill">–</div></div>
</div>

<section>
  <h2><span class="n">1</span>Prisene er ikke Brownsk bevegelse</h2>
  <p>For en random walk er variansratioen VR(q) = Var(X<sub>t+q</sub> − X<sub>t</sub>) / (q · Var(ΔX)) lik 1 på alle horisonter. Her faller den kraftig etter noen timer: prisen trekkes tilbake mot et nivå. Fortegnet på endringene i ubalanseprisen er nesten uforutsigbart, men størrelsen henger sammen over tid, og pristoppene kommer i klynger. Prisene kan også være negative, så heller ikke geometrisk BM passer.</p>
  {vr}
</section>

<section>
  <h2><span class="n">2</span>Kort sikt er forutsigbart, dagen før er spotprisen nesten like god</h2>
  <p>Én time frem ga gradient boosting {skill_rng} % lavere prognosefeil for ubalanseprisen enn naiv persistens, i alle fem prisområdene. Ett døgn frem er ingen modell vesentlig bedre enn å anta at ubalanseprisen blir lik spot (nullmodellen). Budsplitting dagen før handler da mer om risiko enn om forventet pris.</p>
  <h3>Ubalansepris</h3>{bm_pris}
  <h3>Ubalansevolum</h3>{bm_vol}
  <p class="cap">Direkte h-stegs prognoser, test 1.1.–9.10.2026, ny tilpasning hver måned på utvidende vindu. Persistens: verdien blir som nå. Nullmodell: ubalansepris = spot. Sesongnaiv: samme kvarter i går. Klimatologi: snitt av samme kvarter siste 28 døgn. ARX: lineær regresjon. GBM: gradient boosting.</p>
</section>

<section>
  <h2><span class="n">3</span>Hvor forutsigbarheten kommer fra</h2>
  <p>Spreaden mellom ubalansepris og spot og reguleringsretningen henger sammen i noen timer. Volumet er mest persistent. Autokorrelasjonen er rangbasert, fordi noen få ekstreme pristopper ellers skjuler sammenhengen.</p>
  {acf}
  <h3>Døgnmønster i reguleringsretningen</h3><p class="cap">Midt på dagen er det oftest nedregulering, morgen og kveld heller mot oppregulering.</p>{dogn}
  <h3>Volum og pris henger sammen</h3><p class="cap">Overskudd gir nedregulering og ubalansepris under spot, underskudd det motsatte. Søyle = median, strek = 10–90-persentil.</p>{pv}
</section>

<section>
  <h2><span class="n">4</span>Områder og markedsdesign</h2>
  <h3>Samme reguleringsretning</h3><p class="cap">Andel kvarter der to områder har samme reguleringsretning. Sør-Norge (NO1, NO2, NO5) beveger seg nesten som ett område.</p>{samme}
  <h3>Median |ubalansepris − spot| per måned</h3><p class="cap">Overgangene til nordisk mFRR-marked (mars 2025) og 15-minutters spot (oktober 2025) er markert.</p>{mnd}
</section>

<div class="note"><b>Neste steg:</b> sannsynlighetsprognoser for pris og volum, og en LP-modell for budsplitting mellom spot og balansemarkedet med prognosene som input.</div>
<footer>Jeg brukte en AI-assistent til programmeringen og konsentrerte meg om spørsmålene, valg av metode og tolkning av resultatene. Data: <a href="https://opendata.esett.com">eSett Open Data</a>. Kode: <a href="https://github.com/teodorpt03/kortmarkeder">github.com/teodorpt03/kortmarkeder</a>.</footer>
</main>
<script>
const KEYS = {keys};
const AREAS = Object.keys(KEYS);
const seg = document.getElementById('area');
function pick(a) {{
  for (const b of seg.children) b.setAttribute('aria-pressed', String(b.dataset.a === a));
  document.querySelectorAll('[data-k]').forEach(el => el.textContent = KEYS[a][el.dataset.k]);
  document.querySelectorAll('.plotly-graph-div').forEach(gd => {{
    if (!gd.data || !gd.data.some(t => t.meta)) return;
    Plotly.restyle(gd, {{visible: gd.data.map(t => !t.meta || t.meta === a)}});
  }});
}}
AREAS.forEach(a => {{ const b = document.createElement('button'); b.type = 'button'; b.textContent = a; b.dataset.a = a; b.onclick = () => pick(a); seg.appendChild(b); }});
pick('NO1');
</script>
</body></html>
"""


def main():
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(build(), encoding="utf-8")
    print(f"Rapport skrevet til {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
