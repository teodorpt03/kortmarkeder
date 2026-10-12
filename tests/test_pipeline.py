"""Enkle tester for datapipelinen. Kjør: python -m pytest -q

Bruker et ekte svar fra eSett (hentet 10.10.2026) som fasit for formatet, slik at testene
går uten nett.
"""
import pandas as pd

from kortmarkeder import esett, klargjoring as k, lagring

PRICES = [
    {"timestamp": "2026-09-01T02:00:00", "timestampUTC": "2026-09-01T00:00:00Z", "mba": "NO1",
     "imblSalesPrice": 127.30, "imblPurchasePrice": 127.30, "upRegPrice": 129.21, "downRegPrice": 127.30,
     "mainDirRegPowerPerMBA": -1.00, "imblSpotDifferencePrice": -1.91, "upRegPriceFrrA": None,
     "downRegPriceFrrA": None, "valueOfAvoidedActivation": None, "incentivisingComponent": None},
    {"timestamp": "2026-09-01T02:15:00", "timestampUTC": "2026-09-01T00:15:00Z", "mba": "NO1",
     "imblSalesPrice": 126.60, "imblPurchasePrice": 126.60, "upRegPrice": 130.79, "downRegPrice": 126.60,
     "mainDirRegPowerPerMBA": -1.00, "imblSpotDifferencePrice": -4.19, "upRegPriceFrrA": None,
     "downRegPriceFrrA": None, "valueOfAvoidedActivation": None, "incentivisingComponent": None},
]
VOLUMES = [
    {"timestamp": "2026-09-01T02:00:00", "timestampUTC": "2026-09-01T00:00:00Z", "mba": "NO1",
     "imbalance": 13.118940, "imbalancePurchase": 26.355607, "imbalanceSales": 13.236667},
    {"timestamp": "2026-09-01T02:15:00", "timestampUTC": "2026-09-01T00:15:00Z", "mba": "NO1",
     "imbalance": 12.264745, "imbalancePurchase": 27.143377, "imbalanceSales": 14.878632},
]


class FakeResp:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append((url, params))
        return FakeResp(PRICES if "EXP14" in url else VOLUMES)


def test_month_chunks():
    chunks = list(esett._month_chunks(pd.Timestamp("2026-01-15", tz="UTC"), pd.Timestamp("2026-03-02", tz="UTC")))
    assert [(a.strftime("%m-%d"), b.strftime("%m-%d")) for a, b in chunks] == [
        ("01-15", "02-01"), ("02-01", "03-01"), ("03-01", "03-02")]


def test_fetch_parses_and_merges():
    s = FakeSession()
    df = esett.fetch("NO1", "2026-09-01", "2026-09-01 00:30", session=s, verbose=False)
    assert list(df.columns) == esett.ALL_COLS
    assert len(df) == 2
    assert s.calls[0][1]["mba"] == "10YNO_1________2"
    assert df["imbalance"].iloc[0] == 13.118940


def test_prepare_derives_spot_and_spread():
    df = esett.fetch("NO1", "2026-09-01", "2026-09-01 00:30", session=FakeSession(), verbose=False)
    p = k.prepare(df)
    # retning = −1 (ned): ubalansepris = nedreguleringspris, spot = oppreguleringspris (ingen opp-aktivering)
    assert abs(p["spot"].iloc[0] - 129.21) < 1e-9
    assert abs(p["spread"].iloc[1] - (-4.19)) < 1e-9
    assert p["regime"].iloc[0].startswith("R3")
    assert p["time"].iloc[0] == 2  # 00:00 UTC = 02:00 norsk sommertid


def test_merge_prefers_new_values():
    old = pd.DataFrame({"area": ["NO1"], "timestamp_utc": [pd.Timestamp("2026-09-01", tz="UTC")], "x": [1.0]})
    new = pd.DataFrame({"area": ["NO1"], "timestamp_utc": [pd.Timestamp("2026-09-01", tz="UTC")], "x": [2.0]})
    assert lagring.merge(old, new)["x"].tolist() == [2.0]
