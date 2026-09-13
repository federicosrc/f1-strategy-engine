from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

F1_BASE = "https://www.formula1.com"
PIRELLI_BASE = "https://press.pirelli.com"

# Current-season fallback only. This is intentionally not a historical database.
# It mirrors the official Formula 1 2026 calendar and is used only if formula1.com
# is temporarily unreachable from the Streamlit host.
CURRENT_2026_FALLBACK = [
    {"round": 1, "key": "australia", "name": "Australia", "location": "Melbourne", "country": "Australia", "dates": "06–08 Mar"},
    {"round": 2, "key": "china", "name": "China", "location": "Shanghai", "country": "China", "dates": "13–15 Mar"},
    {"round": 3, "key": "japan", "name": "Japan", "location": "Suzuka", "country": "Japan", "dates": "27–29 Mar"},
    {"round": 4, "key": "miami", "name": "Miami", "location": "Miami", "country": "United States", "dates": "01–03 May"},
    {"round": 5, "key": "canada", "name": "Canada", "location": "Montreal", "country": "Canada", "dates": "22–24 May"},
    {"round": 6, "key": "monaco", "name": "Monaco", "location": "Monte Carlo", "country": "Monaco", "dates": "05–07 Jun"},
    {"round": 7, "key": "barcelona-catalunya", "name": "Barcelona-Catalunya", "location": "Barcelona", "country": "Spain", "dates": "12–14 Jun"},
    {"round": 8, "key": "austria", "name": "Austria", "location": "Spielberg", "country": "Austria", "dates": "26–28 Jun"},
    {"round": 9, "key": "great-britain", "name": "Great Britain", "location": "Silverstone", "country": "Great Britain", "dates": "03–05 Jul"},
    {"round": 10, "key": "belgium", "name": "Belgium", "location": "Spa-Francorchamps", "country": "Belgium", "dates": "17–19 Jul"},
    {"round": 11, "key": "hungary", "name": "Hungary", "location": "Budapest", "country": "Hungary", "dates": "24–26 Jul"},
    {"round": 12, "key": "netherlands", "name": "Netherlands", "location": "Zandvoort", "country": "Netherlands", "dates": "21–23 Aug"},
    {"round": 13, "key": "italy", "name": "Italy", "location": "Monza", "country": "Italy", "dates": "04–06 Sep"},
    {"round": 14, "key": "spain", "name": "Spain", "location": "Madrid", "country": "Spain", "dates": "11–13 Sep"},
    {"round": 15, "key": "azerbaijan", "name": "Azerbaijan", "location": "Baku", "country": "Azerbaijan", "dates": "24–26 Sep"},
    {"round": 16, "key": "bahrain", "name": "Bahrain GP in Malaysia", "location": "Kuala Lumpur", "country": "Malaysia", "dates": "02–04 Oct"},
    {"round": 17, "key": "singapore", "name": "Singapore", "location": "Singapore", "country": "Singapore", "dates": "09–11 Oct"},
    {"round": 18, "key": "united-states", "name": "United States", "location": "Austin", "country": "United States", "dates": "23–25 Oct"},
    {"round": 19, "key": "mexico", "name": "Mexico", "location": "Mexico City", "country": "Mexico", "dates": "30 Oct–01 Nov"},
    {"round": 20, "key": "brazil", "name": "Brazil", "location": "Sao Paulo", "country": "Brazil", "dates": "06–08 Nov"},
    {"round": 21, "key": "las-vegas", "name": "Las Vegas", "location": "Las Vegas", "country": "United States", "dates": "19–21 Nov"},
    {"round": 22, "key": "qatar", "name": "Qatar", "location": "Lusail", "country": "Qatar", "dates": "27–29 Nov"},
    {"round": 23, "key": "abu-dhabi", "name": "Abu Dhabi", "location": "Abu Dhabi", "country": "United Arab Emirates", "dates": "04–06 Dec"},
]

# Official Pirelli nominations already announced for the 2026 season as of 2026-09-13.
# No earlier seasons are included.


CURRENT_2026_DRIVERS = [
    "Lando Norris", "Oscar Piastri",
    "George Russell", "Kimi Antonelli",
    "Charles Leclerc", "Lewis Hamilton",
    "Max Verstappen", "Liam Lawson",
    "Pierre Gasly", "Franco Colapinto",
    "Yuki Tsunoda", "Arvid Lindblad",
    "Nico Hulkenberg", "Gabriel Bortoleto",
    "Esteban Ocon", "Oliver Bearman",
    "Alex Albon", "Carlos Sainz",
    "Fernando Alonso", "Lance Stroll",
    "Sergio Perez", "Valtteri Bottas",
]

PIRELLI_2026 = {
    "australia": {"hard": "C3", "medium": "C4", "soft": "C5", "source": "https://press.pirelli.com/complete-f1-tyre-range-for-the-first-three-grands-prix-of-2026/"},
    "china": {"hard": "C2", "medium": "C3", "soft": "C4", "source": "https://press.pirelli.com/complete-f1-tyre-range-for-the-first-three-grands-prix-of-2026/"},
    "japan": {"hard": "C1", "medium": "C2", "soft": "C3", "source": "https://press.pirelli.com/complete-f1-tyre-range-for-the-first-three-grands-prix-of-2026/"},
    "miami": {"hard": "C3", "medium": "C4", "soft": "C5", "source": "https://press.pirelli.com/it/il-tris-piu-morbido-per-le-sfide-di-miami-e-montreal/"},
    "canada": {"hard": "C3", "medium": "C4", "soft": "C5", "source": "https://press.pirelli.com/it/il-tris-piu-morbido-per-le-sfide-di-miami-e-montreal/"},
    "monaco": {"hard": "C3", "medium": "C4", "soft": "C5", "source": "https://press.pirelli.com/the-tyre-compound-selections-for-monte-carlo-and-barcelona/"},
    "barcelona-catalunya": {"hard": "C2", "medium": "C3", "soft": "C4", "source": "https://press.pirelli.com/the-tyre-compound-selections-for-monte-carlo-and-barcelona/"},
    "austria": {"hard": "C3", "medium": "C4", "soft": "C5", "source": "https://press.pirelli.com/it/tutta-la-gamma-pirelli-per-spielberg-e-silverstone/"},
    "great-britain": {"hard": "C1", "medium": "C2", "soft": "C3", "source": "https://press.pirelli.com/it/tutta-la-gamma-pirelli-per-spielberg-e-silverstone/"},
    "belgium": {"hard": "C2", "medium": "C3", "soft": "C4", "source": "https://press.pirelli.com/the-compounds-selected-for-belgium-and-hungary/"},
    "hungary": {"hard": "C3", "medium": "C4", "soft": "C5", "source": "https://press.pirelli.com/the-compounds-selected-for-belgium-and-hungary/"},
    "netherlands": {"hard": "C2", "medium": "C3", "soft": "C4", "source": "https://press.pirelli.com/tyre-compounds-selected-for-zandvoort-monza-and-madrid/"},
    "italy": {"hard": "C3", "medium": "C4", "soft": "C5", "source": "https://press.pirelli.com/tyre-compounds-selected-for-zandvoort-monza-and-madrid/"},
    "spain": {"hard": "C2", "medium": "C3", "soft": "C4", "source": "https://press.pirelli.com/tyre-compounds-selected-for-zandvoort-monza-and-madrid/"},
    "azerbaijan": {"hard": "C3", "medium": "C4", "soft": "C5", "source": "https://press.pirelli.com/tyre-compound-selections-for-baku-sepang-and-singapore/"},
    "bahrain": {"hard": "C2", "medium": "C3", "soft": "C4", "source": "https://press.pirelli.com/tyre-compound-selections-for-baku-sepang-and-singapore/"},
    "singapore": {"hard": "C3", "medium": "C4", "soft": "C5", "source": "https://press.pirelli.com/tyre-compound-selections-for-baku-sepang-and-singapore/"},
}


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1.1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 StrategyEngine/1.2",
        "Accept-Language": "en-GB,en;q=0.9,it;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
    })
    return session


class F1OfficialClient:
    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.session = build_session()

    @property
    def current_year(self) -> int:
        return datetime.now(timezone.utc).year

    def calendar(self) -> list[dict[str, Any]]:
        year = self.current_year
        url = f"{F1_BASE}/en/racing/{year}"
        try:
            r = self.session.get(url, timeout=self.timeout)
            if r.ok:
                events = self._parse_calendar(r.text, year)
                if len(events) >= 15:
                    return events
        except requests.RequestException:
            pass
        if year == 2026:
            return [dict(e, url=f"{F1_BASE}/en/racing/{year}/{e['key']}", source="F1 official fallback snapshot") for e in CURRENT_2026_FALLBACK]
        return []

    def _parse_calendar(self, page: str, year: int) -> list[dict[str, Any]]:
        soup = BeautifulSoup(page, "html.parser")
        found: dict[str, dict[str, Any]] = {}
        pattern = re.compile(rf"^/en/racing/{year}/([^/?#]+)")
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            m = pattern.match(href)
            if not m:
                continue
            key = m.group(1)
            if key in {"testing", "archive"}:
                continue
            text = " ".join(a.stripped_strings)
            if not text or "testing" in text.lower():
                continue
            entry = found.setdefault(key, {
                "key": key,
                "name": key.replace("-", " ").title(),
                "location": key.replace("-", " ").title(),
                "country": key.replace("-", " ").title(),
                "dates": "",
                "url": urljoin(F1_BASE, href),
                "source": "Formula 1 official",
            })
            # Calendar cards normally contain country / event label / dates. Keep the
            # cleanest short label rather than the sponsor-heavy title.
            pieces = [p.strip() for p in re.split(r"\s{2,}|\n", text) if p.strip()]
            if pieces:
                candidate = pieces[0]
                candidate = re.sub(r"^ROUND\s*\d+", "", candidate, flags=re.I).strip()
                if 2 <= len(candidate) <= 45:
                    entry["name"] = candidate
        events = list(found.values())
        # Merge our current-season location/date hints when the key matches. The URLs
        # still come from the live official calendar.
        hints = {e["key"]: e for e in CURRENT_2026_FALLBACK} if year == 2026 else {}
        for i, e in enumerate(events, start=1):
            hint = hints.get(e["key"], {})
            e["round"] = hint.get("round", i)
            e["location"] = hint.get("location", e["location"])
            e["country"] = hint.get("country", e["country"])
            e["dates"] = hint.get("dates", e["dates"])
        return sorted(events, key=lambda x: x.get("round", 999))

    def drivers(self) -> list[str]:
        """Current-season Formula 1 driver list. Official site first, current-season fallback second."""
        try:
            r = self.session.get(f"{F1_BASE}/en/drivers", timeout=self.timeout)
            if r.ok:
                soup = BeautifulSoup(r.text, "html.parser")
                names = []
                seen = set()
                for a in soup.find_all("a", href=True):
                    if not re.match(r"^/en/drivers/[^/?#]+", a.get("href", "")):
                        continue
                    txt = " ".join(a.stripped_strings).strip()
                    # Driver cards may include number/team; extract 2-4 word name candidates.
                    txt = re.sub(r"\b\d{1,2}\b", " ", txt)
                    txt = re.sub(r"\s+", " ", txt).strip()
                    for known in CURRENT_2026_DRIVERS:
                        if known.lower() in txt.lower() and known not in seen:
                            names.append(known); seen.add(known)
                if len(names) >= 18:
                    return names
        except requests.RequestException:
            pass
        return list(CURRENT_2026_DRIVERS)

    def event_details(self, event: dict[str, Any]) -> dict[str, Any]:
        url = event.get("url") or f"{F1_BASE}/en/racing/{self.current_year}/{event['key']}"
        result = dict(event)
        result.setdefault("url", url)
        result["source"] = "Formula 1 official"
        try:
            r = self.session.get(url, timeout=self.timeout)
            if not r.ok:
                return result
            return self._parse_event_page(r.text, result)
        except requests.RequestException:
            return result

    def _parse_event_page(self, page: str, result: dict[str, Any]) -> dict[str, Any]:
        soup = BeautifulSoup(page, "html.parser")
        text = "\n".join(soup.stripped_strings)
        raw = html.unescape(page).replace("\\u002F", "/")

        h1 = soup.find("h1")
        if h1:
            result["event_title"] = " ".join(h1.stripped_strings)

        patterns = {
            "circuit_length_km": r"Circuit Length\s*([0-9]+(?:\.[0-9]+)?)\s*km",
            "number_of_laps": r"Number of Laps\s*([0-9]+)",
            "race_distance_km": r"Race Distance\s*([0-9]+(?:\.[0-9]+)?)\s*km",
            "first_grand_prix": r"First Grand Prix\s*([0-9]{4})",
        }
        for key, pat in patterns.items():
            m = re.search(pat, text, re.I)
            if m:
                val = m.group(1)
                result[key] = int(val) if key in {"number_of_laps", "first_grand_prix"} else float(val)

        # Detailed circuit image from the official Formula 1 race page.
        img_url = None
        for img in soup.find_all("img"):
            attrs = " ".join(str(img.get(k, "")) for k in ["src", "data-src", "srcset", "alt"])
            if re.search(r"track.*detailed|detailed.*track", attrs, re.I):
                src = img.get("src") or img.get("data-src")
                if src:
                    img_url = urljoin(F1_BASE, src)
                    break
        if not img_url:
            candidates = re.findall(r"https?://[^\"'<> ]+(?:track|Track)[^\"'<> ]+(?:detailed|Detailed)[^\"'<> ]+", raw)
            if candidates:
                img_url = candidates[0].replace("&amp;", "&")
        if img_url:
            result["track_image_url"] = img_url

        # Circuit type from current official description only.
        low = text.lower()
        if "semi-permanent" in low or "semi permanent" in low:
            result["circuit_type"] = "Semi-permanent"
        elif "street circuit" in low or "street-circuit" in low or "city streets" in low:
            result["circuit_type"] = "Street"
        else:
            result["circuit_type"] = "Permanent"

        # Race time, if present in the schedule.
        race = re.search(r"(\d{1,2})\s+([A-Z][a-z]{2})\s+Race\s+(\d{1,2}:\d{2})", text)
        if race:
            result["race_schedule_text"] = f"{race.group(1)} {race.group(2)} {race.group(3)}"
        return result


class PirelliCurrentSeason:
    def compounds(self, event_key: str) -> dict[str, Any]:
        row = PIRELLI_2026.get(event_key)
        if row:
            return dict(row, confidence="high", status="Official Pirelli nomination")
        return {
            "hard": "—",
            "medium": "—",
            "soft": "—",
            "source": "https://press.pirelli.com/?h=1&t=2026+tyre+compound+choices",
            "confidence": "pending",
            "status": "Official nomination not bundled yet",
        }
