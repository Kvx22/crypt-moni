"""SEC gets a plain 'crypt-moni <contact>' User-Agent; every other host keeps the project's.
python -m unittest tests.test_research_http_sec"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cryptomoni.research.http import Http, sec_user_agent  # noqa: E402


class _Resp:
    status = 200
    headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=-1):
        return b"{}"


class TestSecUserAgent(unittest.TestCase):
    def seen(self, http, url):
        got = {}

        def fake_open(req, timeout=None):
            got["ua"] = req.get_header("User-agent")
            return _Resp()
        with mock.patch.object(http._opener, "open", fake_open), mock.patch.object(http, "_check", lambda u: u.split("/")[2]):
            http._raw(url, url.split("/")[2], check=False)
        return got["ua"]

    def test_format(self):
        self.assertEqual(sec_user_agent(" me@example.com "), "crypt-moni me@example.com")
        self.assertIsNone(sec_user_agent(None))
        self.assertIsNone(sec_user_agent(""))

    def test_only_sec_hosts_get_it(self):
        h = Http(user_agent="long-ua contact:me@example.com", sec_contact="me@example.com")
        self.assertEqual(self.seen(h, "https://www.sec.gov/files/company_tickers.json"), "crypt-moni me@example.com")
        self.assertEqual(self.seen(h, "https://data.sec.gov/submissions/CIK0000320193.json"), "crypt-moni me@example.com")
        self.assertEqual(self.seen(h, "https://example.com/x"), "long-ua contact:me@example.com")
        self.assertEqual(self.seen(h, "https://notsec.gov.evil.com/x"), "long-ua contact:me@example.com")

    def test_no_contact_changes_nothing(self):
        h = Http(user_agent="long-ua")
        self.assertEqual(self.seen(h, "https://www.sec.gov/x"), "long-ua")


if __name__ == "__main__":
    unittest.main()
