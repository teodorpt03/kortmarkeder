"""Lokal lagring og løpende oppdatering av datagrunnlaget.

Alt lagres i én parquet-fil (data/esett_15min.parquet) med én rad per (område, 15-min MTU).
Oppdatering er inkrementell: for hvert område hentes data fra siste lagrede tidspunkt minus
noen dagers overlapp (eSett kan korrigere tall i etterkant) og frem til nå.

Bruk fra terminalen:
    python -m kortmarkeder.lagring                      # oppdater alle områder til nå
    python -m kortmarkeder.lagring --importer data/esett_no_2024_2026.csv.gz
    python -m kortmarkeder.lagring --fra 2024-01-01     # full nedlasting fra en dato
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import AREAS, esett

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PARQUET = DATA_DIR / "esett_15min.parquet"
DEFAULT_START = "2024-01-01"


def load(path: Path = PARQUET) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Fant ikke {path}. Kjør først `python -m kortmarkeder.lagring` "
            "(eller --importer med csv.gz-filen)."
        )
    return pd.read_parquet(path)


def save(df: pd.DataFrame, path: Path = PARQUET) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["area", "timestamp_utc"]).reset_index(drop=True)
    df.to_parquet(path, index=False)


def merge(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Slå sammen gammelt og nytt; nye verdier vinner ved overlapp (korrigeringer)."""
    if old is None or old.empty:
        return new.drop_duplicates(["area", "timestamp_utc"], keep="last")
    both = pd.concat([old, new], ignore_index=True)
    return both.drop_duplicates(["area", "timestamp_utc"], keep="last")


def import_csv(csv_path: str | Path) -> pd.DataFrame:
    """Les inn en eksport (samme kolonner som esett.ALL_COLS) og lagre som parquet."""
    df = pd.read_csv(csv_path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df[esett.ALL_COLS]
    old = pd.read_parquet(PARQUET) if PARQUET.exists() else None
    out = merge(old, df)
    save(out)
    return out


def update(areas=AREAS, start: str | None = None, overlap_days: int = 3,
           end=None, verbose: bool = True) -> pd.DataFrame:
    """Hent nye data for hvert område og skriv til parquet. Returnerer hele datasettet."""
    old = pd.read_parquet(PARQUET) if PARQUET.exists() else None
    end = esett.to_utc(end) if end is not None else esett.now_utc()
    new_parts = []
    for area in areas:
        if start is not None:
            a = esett.to_utc(start)
        elif old is not None and (old["area"] == area).any():
            last = old.loc[old["area"] == area, "timestamp_utc"].max()
            a = (last - pd.Timedelta(days=overlap_days)).floor("D")
        else:
            a = esett.to_utc(DEFAULT_START)
        if verbose:
            print(f"Henter {area} fra {a:%Y-%m-%d %H:%M} til {end:%Y-%m-%d %H:%M} UTC")
        new_parts.append(esett.fetch(area, a, end, verbose=verbose))
    new = pd.concat(new_parts, ignore_index=True)
    out = merge(old, new)
    save(out)
    if verbose:
        n_new = len(out) - (0 if old is None else len(old))
        print(f"Ferdig: {len(out):,} rader totalt ({n_new:+,} nye). Lagret i {PARQUET}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Oppdater datagrunnlaget fra eSett.")
    p.add_argument("--omraader", nargs="+", default=AREAS, help="f.eks. NO1 NO2")
    p.add_argument("--fra", default=None, help="startdato (ellers fra siste lagrede punkt)")
    p.add_argument("--importer", default=None, help="importer en csv(.gz)-eksport i stedet")
    args = p.parse_args(argv)
    if args.importer:
        df = import_csv(args.importer)
        print(f"Importerte {len(df):,} rader til {PARQUET}")
    else:
        update(args.omraader, start=args.fra)


if __name__ == "__main__":
    main()
