# Prediksjon av pris og volum i korte kraftmarkeder – miniprosjekt

Et lite forprosjekt laget i forbindelse med søknad til sommerforskerprosjektet *«Datadrevet prediksjon av pris og volum i korte kraftmarkeder»* ved SINTEF Energi. Prosjektet følger arbeidsoppgavene i utlysningen:

| Utlysningen ber om | Her |
|---|---|
| Sette seg inn i datagrunnlaget for prediksjon av priser og volumer | `kortmarkeder/klargjoring.py`, `kortmarkeder/analyse.py` |
| Løsninger for automatisk datainnhenting og visualisering | `kortmarkeder/esett.py`, `kortmarkeder/lagring.py`, `oppdater.sh`, `dashboard.py` |
| Implementering av (naive) benchmark-modeller fra litteraturen | `kortmarkeder/benchmark.py` |
| Utprøving av egne modeller | ARX (ridge) og gradient boosting i `benchmark.py` |

## Datagrunnlag

**eSett Open Data** (`api.opendata.esett.com`) er et åpent API uten nøkkel fra det nordiske ubalanseoppgjøret. Prosjektet bruker:

- `EXP14/Prices` – ubalansepris, opp- og nedreguleringspris, hovedretning for regulering og differansen mellom ubalansepris og spot
- `EXP13/ImbalancePowerVolume` – avregnet ubalansevolum

Alle fem norske budområder (NO1–NO5), 15-minutters oppløsning, fra januar 2024. Spotprisen avledes som `ubalansepris − imblSpotDifferencePrice`. Det er kontrollert mot opp- og nedreguleringsprisene.

Dataene viser tre markedsdesign-regimer, og de må tas hensyn til ved modellering:

| Regime | Periode | Oppløsning |
|---|---|---|
| R1 | før 4.3.2025 | ubalansepris i praksis per time |
| R2 | 4.3.2025–30.9.2025 | nordisk mFRR-energiaktiveringsmarked: ubalansepris per kvarter, spot per time |
| R3 | fra 1.10.2025 | 15-minutters day-ahead: hele kjeden per kvarter |

## Kom i gang

Kjør alle kommandoene fra prosjektmappen, altså mappen der denne README-en, `dashboard.py` og `requirements.txt` ligger. Undermappen `kortmarkeder/` er Python-pakken, og dit trenger du ikke gå.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m kortmarkeder.lagring
python -m kortmarkeder.analyse
python -m kortmarkeder.benchmark
python -m kortmarkeder.eksport
streamlit run dashboard.py
python -m pytest
```

| Kommando | Hva den gjør |
|---|---|
| `kortmarkeder.lagring` | henter nye data fra eSett (inkrementelt) → `data/esett_15min.parquet` |
| `kortmarkeder.analyse` | stiliserte fakta → `results/analyse.json` |
| `kortmarkeder.benchmark` | benchmark-modeller (ca. 15 min for alle områder) → `results/benchmark.*` |
| `kortmarkeder.eksport` | flate CSV-tabeller → `results/tabeller/` |
| `streamlit run dashboard.py` | dashbordet i nettleseren |
| `pytest` | tester (går uten nett) |

`data/esett_15min.parquet` ligger allerede ved siden av koden, så analysen og dashbordet virker uten nedlasting. `lagring` henter bare det som er nytt, med tre dagers overlapp fordi eSett kan korrigere tall i etterkant.

**Løpende oppdatering:** `oppdater.sh` henter nye data og regner ut analysen på nytt. Den kan kjøres daglig med cron (se kommentaren i filen). Dashbordet har også en knapp, «Oppdater data nå».

## Struktur

```
kortmarkeder/
  esett.py         klient mot eSett-API-et (månedsvise biter, retry)
  lagring.py       parquet-lagring, inkrementell oppdatering, import av CSV
  klargjoring.py   avledede størrelser (spot, spread, retning), kalender og regimer
  analyse.py       stiliserte fakta
  benchmark.py     prognosemodeller og rullerende evaluering
  eksport.py       resultater som CSV
dashboard.py       Streamlit-dashboard
tests/             tester for parsing og klargjøring
results/           ferdige resultater (analyse.json, benchmark.csv, tabeller/)
```

## Spørsmål og funn

Analyseperioden er R3 (1.10.2025–9.10.2026). Tallene gjelder NO1 der ikke annet står.

**1. Er prisene Brownsk bevegelse?** Nei.
- Variansratioen VR(q) faller fra ca. 1 til 0,18 ved ett døgn og 0,04 ved én uke. Prisen trekkes kraftig tilbake mot et nivå. Halveringstiden for avvik i døgnsnittet av spot er ca. 1,6 døgn.
- I NO1, NO2 og NO5 er VR over 1 rundt 4 timer (1,26 i NO1). Spotprisen har altså trend innen døgnrampene.
- Kurtosen i kvartersendringene er 36 for spot og over 10 000 for ubalanseprisen, mot 0 for normalfordeling. ADF forkaster enhetsrot (p < 0,001).
- Prisene kan være negative, så geometrisk BM og log-avkastning passer ikke. Mer naturlige modeller er Ornstein–Uhlenbeck med sesong og hopp, eller regimebytte.

**2. Er de forutsigbare (autokorrelasjon)?** På kort sikt, ja.
- Den rangbaserte autokorrelasjonen i spreaden (ubalansepris − spot) er 0,83 ved ett kvarter, 0,64 ved én time og 0,18 ved ett døgn. Uten rangering er den bare 0,53 ved ett kvarter, fordi noen få pristopper skjuler sammenhengen.
- Reguleringsretningen er en Markov-kjede der P(samme retning neste kvarter) er ca. 0,85. En retning varer i snitt ca. 1¾ time.
- Pristopper kommer i klynger. Gitt en topp er sannsynligheten for en ny topp innen en time ca. 63 %, mot 1 % ellers. Det likner GARCH-effekter i finans.

**3. Kalender- eller hendelsestid?** For ubalanse er kalendertid (leveringskvarteret) det naturlige, fordi hvert kvarter avregnes for seg. Retningen har et klart døgnmønster: midt på dagen er det oftest nedregulering og ubalansepris under spot, mens morgen og kveld heller mot oppregulering. Hendelsestid blir relevant for intradagmarkedet med kontinuerlig handel (krever handelsdata).

**4. Er volumet forutsigbart?** Ja, det er den mest persistente størrelsen (autokorrelasjon 0,89 ved én time). Volumet forklarer også prisen. Overskudd gir nedregulering og ubalansepris under spot, underskudd gir det motsatte, og kurven har en lang øvre hale. Det likner en tilbudskurve for balansekraft.

**5. Hendelser mot hverandre:** NO1, NO2 og NO5 har samme reguleringsretning i 83–92 % av kvarterene. NO3 og NO4 beveger seg mer for seg selv (55–66 %).

**Benchmark** (test 1.1–9.10.2026, månedlig re-tilpasning):

| Ubalansepris, MAE (EUR/MWh), NO1 | 15 min | 1 time | 4 timer | 1 døgn |
|---|---|---|---|---|
| Persistens | 5,81 | 9,88 | 13,85 | 15,23 |
| Nullmodell (ubalansepris = spot) | 10,22 | 10,22 | 10,22 | 10,22 |
| Klimatologi | 11,97 | 11,97 | 11,97 | 11,97 |
| ARX (ridge) | 9,32 | 10,43 | 11,33 | 10,75 |
| **GBM** | **5,68** | **8,23** | **9,80** | 10,23 |

- Én time frem slår GBM persistens med 17–25 % i alle områder.
- Ett døgn frem er ingen modell vesentlig bedre enn å bruke spotprisen. På budtidspunktet dagen før er spot altså nær den beste punktprognosen for ubalanseprisen, og verdien ligger i å modellere risikoen (fordelingen).
- Den lineære modellen er tilpasset kvadratfeil og taper mot persistens på MAE ved kort horisont, fordi spreaden er skjev og ofte nøyaktig 0. Valget av tapsfunksjon betyr mye.
- Reguleringsretning: 85–88 % treff ett kvarter frem og 75–81 % én time frem, men bare 58–66 % ett døgn frem.

## Forbehold

- Prognosen antar at data for kvarter t er kjent ved t. I drift kommer dataene med forsinkelse, så reell horisont blir noe lengre.
- Sammenligningen før og etter 15-minutters spot er en observasjon, ikke en påvist årsak. Prisnivå og kraftbalanse har også endret seg.
- Bare én datakilde. Vær-, forbruks- og flytdata ville trolig forbedre volumprognosene.

## Neste steg

1. Sannsynlighetsprognoser (kvantilregresjon, pinball-tap) for spread og volum.
2. Flere forklaringsvariabler: vind- og forbruksprognoser (ENTSO-E), flyt mellom områder, intradagpriser.
3. En enkel LP-modell for budsplitting mellom spot og balansemarkedet, med prognosene som input. Det er det langsiktige målet i prosjektet.
