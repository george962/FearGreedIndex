import unittest
from datetime import datetime, timezone

from FearGreed import parse_record


class ParseRecordTests(unittest.TestCase):
    def test_parses_current_record(self):
        record = {
            "score": 39.4285714285714,
            "rating": "fear",
            "timestamp": "2026-07-24T23:59:54+00:00",
        }

        self.assertEqual(
            parse_record(record),
            (
                "2026-07-24",
                39.4285714285714,
                "fear",
                datetime(2026, 7, 24, 23, 59, 54, tzinfo=timezone.utc),
            ),
        )

    def test_accepts_z_timestamp(self):
        record = {
            "score": "51.6",
            "rating": "neutral",
            "timestamp": "2026-07-25T20:00:00Z",
        }

        self.assertEqual(
            parse_record(record),
            (
                "2026-07-25",
                51.6,
                "neutral",
                datetime(2026, 7, 25, 20, 0, tzinfo=timezone.utc),
            ),
        )

    def test_rejects_unexpected_response(self):
        with self.assertRaisesRegex(ValueError, "unexpected data format"):
            parse_record({})


if __name__ == "__main__":
    unittest.main()