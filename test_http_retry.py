import io
import unittest
from urllib.error import HTTPError
from urllib.request import Request

from http_retry import fetch_json_with_retry


class JsonResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class HttpRetryTests(unittest.TestCase):
    def test_retries_transient_http_error(self):
        calls = []
        sleeps = []

        def opener(*_args, **_kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise HTTPError("https://example.test", 503, "busy", {}, None)
            return JsonResponse(b'{"ok": true}')

        result = fetch_json_with_retry(
            Request("https://example.test"),
            timeout=1,
            context=None,
            attempts=2,
            base_delay_seconds=0.0,
            opener=opener,
            sleep=sleeps.append,
        )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(sleeps), 1)

    def test_does_not_retry_non_transient_http_error(self):
        calls = []

        def opener(*_args, **_kwargs):
            calls.append(1)
            raise HTTPError("https://example.test", 404, "missing", {}, None)

        with self.assertRaisesRegex(RuntimeError, "404"):
            fetch_json_with_retry(
                Request("https://example.test"),
                timeout=1,
                context=None,
                attempts=4,
                opener=opener,
                sleep=lambda _seconds: None,
            )
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
