import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.build_dashboard import (
    add_features,
    merge_signals,
    parse_combined_dataset,
    parse_fear_dataset,
    parse_market_dataset,
)


class DashboardDataTests(unittest.TestCase):
    def test_parses_repository_combined_schema(self):
        content = """fear_greed_date_utc,fear_greed_value,spx_date,spx_open,spx_high,spx_low,spx_close\n2024-01-06,20,2024-01-05,4700,4720,4680,4710\n2024-01-08,25,2024-01-08,4725,4750,4710,4740\n"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "combined.csv"
            path.write_text(content, encoding="utf-8")
            daily, market = parse_combined_dataset(path)

        self.assertEqual(list(daily["fear_greed"]), [20, 25])
        self.assertEqual(list(market.columns), ["open", "high", "low", "close"])
        self.assertEqual(float(market.loc[pd.Timestamp("2024-01-05"), "low"]), 4680)

    def test_parses_separate_repository_schemas_and_enters_next_session(self):
        fear_content = """Date,Value,Rating,Source Timestamp UTC\n2024-01-06,20,fear,2024-01-06T00:00:00Z\n2024-01-08,25,fear,2024-01-08T00:00:00Z\n"""
        market_content = """date,symbol,open,high,low,close,adj_close,volume,data_source\n2024-01-05,^GSPC,4700,4720,4680,4710,4710,1,test\n2024-01-08,^GSPC,4725,4750,4710,4740,4740,1,test\n2024-01-09,^GSPC,4730,4760,4720,4750,4750,1,test\n2024-01-10,^GSPC,4755,4770,4740,4760,4760,1,test\n2024-01-11,^GSPC,4760,4780,4750,4770,4770,1,test\n2024-01-12,^GSPC,4770,4790,4760,4780,4780,1,test\n"""
        with tempfile.TemporaryDirectory() as directory:
            fear_path = Path(directory) / "fear.csv"
            market_path = Path(directory) / "market.csv"
            fear_path.write_text(fear_content, encoding="utf-8")
            market_path.write_text(market_content, encoding="utf-8")
            daily = parse_fear_dataset(fear_path)
            market = parse_market_dataset(market_path)
            daily, market = add_features(daily, market)
            merged = merge_signals(daily, market)

        weekend = merged.loc[merged["signal_date"] == pd.Timestamp("2024-01-06")].iloc[0]
        self.assertEqual(weekend["market_date"], pd.Timestamp("2024-01-05"))
        self.assertEqual(weekend["entry_date"], pd.Timestamp("2024-01-08"))
        self.assertEqual(float(weekend["entry_price"]), 4725)


if __name__ == "__main__":
    unittest.main()