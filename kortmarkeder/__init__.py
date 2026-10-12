"""Datadrevet prediksjon av pris og volum i korte kraftmarkeder (miniprosjekt).

Moduler:
    esett       - klient mot eSett Open Data API (ubalansepriser og -volumer, NO1-NO5)
    lagring     - lokal parquet-lagring med inkrementell oppdatering
    klargjoring - rensing, avledede størrelser (spot, spread) og kalendervariabler
    analyse     - stiliserte fakta: Brownsk bevegelse?, autokorrelasjon, kalender, volum
    benchmark   - naive/sesongnaive/klimatologi/AR-modeller med rullerende evaluering
"""

AREAS = ["NO1", "NO2", "NO3", "NO4", "NO5"]
