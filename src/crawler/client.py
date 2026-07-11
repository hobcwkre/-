"""Shared HTTP client for TPEx (Taipei Exchange) data endpoints.

Two kinds of endpoints are used:
  - OpenAPI (https://www.tpex.org.tw/openapi/v1/...): simple GET, JSON list, no params.
    Used for the OTC / emerging company universe.
  - Legacy "www" query endpoints (https://www.tpex.org.tw/www/zh-tw/...): POST,
    form-encoded, used for historical daily quotes. These are not officially
    documented; they were reverse-engineered from the public website's own
    network calls, so keep requests polite (delay + a real User-Agent).
"""
from __future__ import annotations

import ssl
import time
from datetime import date

import requests

BASE = "https://www.tpex.org.tw"
OPENAPI_BASE = f"{BASE}/openapi/v1"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Referer": f"{BASE}/zh-tw/",
}


class _RelaxedSSLAdapter(requests.adapters.HTTPAdapter):
    """TPEx's cert lacks a Subject Key Identifier extension, which recent
    OpenSSL builds (3.2+, the X509_V_FLAG_X509_STRICT check) reject even
    though the chain and hostname are otherwise valid. This adapter drops
    only that one strict-mode flag; full chain and hostname verification
    still applies.
    """

    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        if hasattr(ssl, "VERIFY_X509_STRICT"):
            ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


class TpexClient:
    def __init__(self, delay: float = 0.4, timeout: float = 15.0):
        self.delay = delay
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(_HEADERS)
        self.session.mount("https://", _RelaxedSSLAdapter())

    def get_openapi(self, endpoint: str) -> list[dict]:
        url = f"{OPENAPI_BASE}/{endpoint}"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        time.sleep(self.delay)
        return resp.json()

    def post_query(self, action: str, data: dict) -> dict:
        """POST to /www/zh-tw/{action} with form-encoded data, return decoded JSON."""
        url = f"{BASE}/www/zh-tw/{action}"
        resp = self.session.post(url, data=data, timeout=self.timeout)
        resp.raise_for_status()
        time.sleep(self.delay)
        return resp.json()


def to_query_date(d: date) -> str:
    """Format a date the way TPEx's www query endpoints expect: AD, slash-separated."""
    return f"{d.year}/{d.month:02d}/{d.day:02d}"
