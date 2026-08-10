"""SEC EDGAR client.

EDGAR is free and unauthenticated, which makes it easy to abuse by accident. The
SEC's published terms require a descriptive User-Agent with contact information
and cap clients at 10 requests/second; exceeding that earns an IP block that
affects everyone behind it. Both rules are enforced here rather than left to
callers, because "remember to throttle" is not a control.

Two other things belong at this layer:

* **On-disk caching of raw documents.** A 10-K is ~1-10 MB and immutable once
  filed. Re-downloading it during development is slow and rude; caching makes
  re-runs offline-capable and makes the ingest step reproducible.
* **Retry on transient failure only.** A 404 on a filing is a data problem to
  surface, not something to hammer.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import httpx

from ..settings import get_settings

log = logging.getLogger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"

#: Forms this project extracts signals from. Anything else is ignored at ingest
#: rather than filtered later, to keep the warehouse focused.
SUPPORTED_FORMS = ("10-K", "10-Q", "8-K")


class EdgarError(RuntimeError):
    pass


@dataclass(slots=True)
class Company:
    cik: str  # zero-padded 10-digit
    name: str
    ticker: str | None = None
    sic: str | None = None
    sic_description: str | None = None
    exchange: str | None = None


@dataclass(slots=True)
class Filing:
    accession: str  # 0000320193-26-000020
    cik: str
    form: str
    filing_date: date
    report_date: date | None
    primary_doc: str
    items: str = ""  # 8-K item codes, comma separated
    size_bytes: int | None = None

    @property
    def accession_nodash(self) -> str:
        return self.accession.replace("-", "")

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(
            cik_int=int(self.cik),
            acc_nodash=self.accession_nodash,
            doc=self.primary_doc,
        )


class RateLimiter:
    """Thread-safe minimum-interval limiter.

    Deliberately a simple spacing limiter rather than a token bucket: a bucket
    permits a burst that would breach the SEC's per-second cap, which is exactly
    the behaviour we are trying to prevent.
    """

    def __init__(self, per_second: float) -> None:
        self.min_interval = 1.0 / per_second if per_second > 0 else 0.0
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            elapsed = time.monotonic() - self._last
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last = time.monotonic()


class EdgarClient:
    def __init__(
        self,
        *,
        user_agent: str | None = None,
        rate_limit_per_s: float | None = None,
        cache_dir: Path | None = None,
        max_retries: int = 3,
    ) -> None:
        s = get_settings()
        self.user_agent = user_agent or s.sec_user_agent
        if "@" not in self.user_agent:
            # The SEC asks for contact info; a UA without it risks a block that
            # would look like a mysterious network failure to whoever hits it.
            log.warning(
                "SEC user agent %r has no contact email; set SF_SEC_USER_AGENT",
                self.user_agent,
            )
        self.limiter = RateLimiter(rate_limit_per_s or s.sec_rate_limit_per_s)
        self.cache_dir = cache_dir or (s.data_dir / "edgar_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self._client = httpx.Client(
            headers={
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=60.0,
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- HTTP --------------------------------------------------------------
    def _get(self, url: str, *, cache: bool = True) -> bytes:
        cache_path = self.cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()[:24]}.bin"
        if cache and cache_path.exists():
            return cache_path.read_bytes()

        last: Exception | None = None
        for attempt in range(self.max_retries):
            self.limiter.wait()
            try:
                resp = self._client.get(url)
            except httpx.HTTPError as exc:
                last = exc
            else:
                if resp.status_code == 200:
                    if cache:
                        cache_path.write_bytes(resp.content)
                    return resp.content
                if resp.status_code in (403, 404):
                    # Permanent: retrying wastes quota and hides the real cause.
                    raise EdgarError(f"EDGAR {resp.status_code} for {url}")
                last = EdgarError(f"EDGAR {resp.status_code} for {url}")
            if attempt < self.max_retries - 1:
                time.sleep(2.0**attempt)
        raise EdgarError(f"EDGAR request failed after {self.max_retries} attempts: {last}")

    # -- metadata ----------------------------------------------------------
    def resolve_ticker(self, ticker: str) -> str:
        """Map a ticker to a zero-padded CIK."""
        data = _json(self._get(TICKER_MAP_URL))
        want = ticker.upper().strip()
        for row in data.values():
            if str(row.get("ticker", "")).upper() == want:
                return f"{int(row['cik_str']):010d}"
        raise EdgarError(f"ticker {ticker!r} not found in SEC company_tickers.json")

    def company(self, cik_or_ticker: str) -> Company:
        cik = self._normalise_cik(cik_or_ticker)
        data = _json(self._get(SUBMISSIONS_URL.format(cik=cik), cache=False))
        tickers = data.get("tickers") or []
        exchanges = data.get("exchanges") or []
        return Company(
            cik=cik,
            name=data.get("name", ""),
            ticker=tickers[0] if tickers else None,
            sic=data.get("sic"),
            sic_description=data.get("sicDescription"),
            exchange=exchanges[0] if exchanges else None,
        )

    def filings(
        self,
        cik_or_ticker: str,
        *,
        forms: tuple[str, ...] = SUPPORTED_FORMS,
        since: date | None = None,
        limit: int | None = None,
    ) -> list[Filing]:
        """Recent filings for a company, newest first.

        Only the ``filings.recent`` block is read (roughly the last 1,000 filings
        or one year). Older filings live in paginated overflow files; that is a
        deliberate scope boundary — signals here are about *recent* disclosure,
        and the full history would multiply ingest cost for no gain.
        """
        cik = self._normalise_cik(cik_or_ticker)
        data = _json(self._get(SUBMISSIONS_URL.format(cik=cik), cache=False))
        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            return []

        out: list[Filing] = []
        cols = (
            recent["accessionNumber"],
            recent["form"],
            recent["filingDate"],
            recent.get("reportDate") or [""] * len(recent["form"]),
            recent.get("primaryDocument") or [""] * len(recent["form"]),
            recent.get("items") or [""] * len(recent["form"]),
            recent.get("size") or [None] * len(recent["form"]),
        )
        for acc, form, fdate, rdate, doc, items, size in zip(*cols, strict=False):
            if form not in forms or not doc:
                continue
            filed = _parse_date(fdate)
            if filed is None or (since and filed < since):
                continue
            out.append(
                Filing(
                    accession=acc,
                    cik=cik,
                    form=form,
                    filing_date=filed,
                    report_date=_parse_date(rdate),
                    primary_doc=doc,
                    items=items or "",
                    size_bytes=int(size) if size else None,
                )
            )
            if limit and len(out) >= limit:
                break
        return out

    def document(self, filing: Filing) -> str:
        """Fetch a filing's primary document as text (usually inline-XBRL HTML)."""
        raw = self._get(filing.url)
        return raw.decode("utf-8", errors="replace")

    def iter_documents(self, filings: list[Filing]) -> Iterator[tuple[Filing, str]]:
        """Yield documents one at a time, skipping the ones that fail.

        A single missing document must not abort a 200-filing backfill; the skip
        is logged so it shows up rather than vanishing.
        """
        for f in filings:
            try:
                yield f, self.document(f)
            except EdgarError as exc:
                log.warning("skipping %s %s: %s", f.form, f.accession, exc)

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _normalise_cik(value: str) -> str:
        value = value.strip()
        digits = re.sub(r"\D", "", value)
        if digits and (digits == value or value.upper().startswith("CIK")):
            return f"{int(digits):010d}"
        if digits and len(digits) >= 6:
            return f"{int(digits):010d}"
        raise EdgarError(
            f"{value!r} is not a CIK; resolve tickers with EdgarClient.resolve_ticker first"
        )

    def health(self) -> tuple[bool, str]:
        try:
            self.company("0000320193")
        except Exception as exc:
            return False, f"edgar unreachable: {exc}"
        return True, "edgar ok"


def _json(raw: bytes) -> dict:
    import json

    return json.loads(raw.decode("utf-8", errors="replace"))


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None
