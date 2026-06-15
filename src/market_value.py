"""
Transfermarkt squad market values for WC 2026 national teams.

Fetches via cloudscraper (handles Cloudflare JS challenges).
Results cached to data/cache/market_values.csv.

Fallback table: hand-curated June 2026 estimates in EUR-M from Transfermarkt.
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path

import pandas as pd

CACHE_PATH = Path(__file__).parent.parent / "data" / "cache" / "market_values.csv"

# Transfermarkt national team page: (url_slug, numeric_id)
_TM_TEAMS: dict[str, tuple[str, int]] = {
    "England":                 ("england",              3301),
    "France":                  ("frankreich",           3377),
    "Spain":                   ("spanien",              3375),
    "Germany":                 ("deutschland",          3376),
    "Brazil":                  ("brasilien",            3439),
    "Argentina":               ("argentinien",          3437),
    "Portugal":                ("portugal",             3380),
    "Netherlands":             ("niederlande",          3378),
    "Belgium":                 ("belgien",              3382),
    "Japan":                   ("japan",                3468),
    "Colombia":                ("kolumbien",            3446),
    "Norway":                  ("norwegen",             3386),
    "Mexico":                  ("mexiko",               3413),
    "United States":           ("vereinigte-staaten",   3411),
    "Switzerland":             ("schweiz",              3388),
    "Croatia":                 ("kroatien",             3190),
    "Uruguay":                 ("uruguay",              3443),
    "Ecuador":                 ("ecuador",              3447),
    "Australia":               ("australien",           3456),
    "Morocco":                 ("marokko",              3474),
    "Senegal":                 ("senegal",              3476),
    "Turkey":                  ("tuerkei",              3383),
    "South Korea":             ("suedkorea",            3469),
    "Iran":                    ("iran",                 3463),
    "Paraguay":                ("paraguay",             3444),
    "Austria":                 ("oesterreich",          3393),
    "Algeria":                 ("algerien",             3475),
    "Czech Republic":          ("tschechien",           3392),
    "Scotland":                ("schottland",           3389),
    "Sweden":                  ("schweden",             3387),
    "Canada":                  ("kanada",               3441),
    "Ivory Coast":             ("elfenbeinkueste",      3477),
    "Ghana":                   ("ghana",                3478),
    "Egypt":                   ("aegypten",             3479),
    "DR Congo":                ("dr-kongo",             3481),
    "Cape Verde":              ("kap-verde",            3480),
    "Saudi Arabia":            ("saudi-arabien",        3464),
    "Iraq":                    ("irak",                 3467),
    "Jordan":                  ("jordanien",            3466),
    "Uzbekistan":              ("usbekistan",           3472),
    "New Zealand":             ("neuseeland",           3459),
    "Panama":                  ("panama",               3412),
    "Haiti":                   ("haiti",                3414),
    "Tunisia":                 ("tunesien",             3473),
    "South Africa":            ("suedafrika",           3482),
    "Bosnia and Herzegovina":  ("bosnien-herzegowina",  3391),
    "Qatar":                   ("katar",                3471),
    "Curacao":                 ("curacao",              7887),
    "Curaçao":                 ("curacao",              7887),
}

# Fallback values (EUR-M) — Transfermarkt squad totals, June 2026 estimates.
_FALLBACK_VALUES: dict[str, float] = {
    "England":                  1180.0,
    "France":                   1090.0,
    "Spain":                    1060.0,
    "Germany":                   930.0,
    "Brazil":                    850.0,
    "Portugal":                  780.0,
    "Netherlands":               730.0,
    "Belgium":                   620.0,
    "Argentina":                 600.0,
    "Colombia":                  430.0,
    "Norway":                    380.0,
    "Japan":                     350.0,
    "Croatia":                   320.0,
    "Switzerland":               300.0,
    "Mexico":                    285.0,
    "Australia":                 270.0,
    "United States":             265.0,
    "Turkey":                    260.0,
    "South Korea":               255.0,
    "Austria":                   210.0,
    "Sweden":                    195.0,
    "Scotland":                  190.0,
    "Uruguay":                   185.0,
    "Ecuador":                   180.0,
    "Senegal":                   175.0,
    "Morocco":                   170.0,
    "Czech Republic":            160.0,
    "Canada":                    155.0,
    "Paraguay":                  130.0,
    "Ivory Coast":               125.0,
    "Algeria":                   120.0,
    "Iran":                      115.0,
    "Ghana":                     110.0,
    "Tunisia":                    95.0,
    "Egypt":                      90.0,
    "Bosnia and Herzegovina":     85.0,
    "Saudi Arabia":               80.0,
    "Iraq":                       60.0,
    "Jordan":                     55.0,
    "DR Congo":                   50.0,
    "Uzbekistan":                 45.0,
    "Panama":                     45.0,
    "Cape Verde":                 40.0,
    "New Zealand":                38.0,
    "South Africa":               35.0,
    "Qatar":                      30.0,
    "Haiti":                      25.0,
    "Curaçao":                    12.0,
    "Curacao":                    12.0,
}
_DEFAULT_MV = 100.0


def _fetch_squad_value(slug: str, team_id: int, timeout: int = 15) -> float | None:
    """Scrape squad market value from Transfermarkt page."""
    try:
        import cloudscraper
        scraper = cloudscraper.create_scraper()
        url = f"https://www.transfermarkt.com/{slug}/startseite/verein/{team_id}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        r = scraper.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return None

        # Match patterns like "€1.18bn", "€930.00m", "€45.50m"
        m = re.search(
            r'class="data-header__market-value-wrapper"[^>]*>.*?'
            r'€\s*([\d,.]+)\s*(bn|m|k)',
            r.text, re.DOTALL | re.IGNORECASE,
        )
        if not m:
            return None

        raw = float(m.group(1).replace(",", ""))
        unit = m.group(2).lower()
        if unit == "bn":
            raw *= 1000
        elif unit == "k":
            raw /= 1000
        value = round(raw, 1)
        # Sanity check: national team squads are worth at minimum a few million EUR
        if value < 5.0:
            return None
        return value

    except Exception:
        return None


def load_market_values(force: bool = False) -> dict[str, float]:
    """
    Return squad market values (EUR-M) for all WC 2026 teams.

    Tries a live Transfermarkt scrape for top teams first; fills the rest
    from the hardcoded fallback table.
    """
    if CACHE_PATH.exists() and not force:
        df = pd.read_csv(CACHE_PATH)
        return dict(zip(df["team"], df["market_value_eur_m"]))

    values: dict[str, float] = {}

    # Live scrape — attempt for the highest-value squads where accuracy matters most
    priority = [
        "England", "France", "Spain", "Germany", "Brazil",
        "Portugal", "Netherlands", "Belgium", "Argentina",
    ]
    for team in priority:
        info = _TM_TEAMS.get(team)
        if info:
            v = _fetch_squad_value(info[0], info[1])
            fallback = _FALLBACK_VALUES.get(team, _DEFAULT_MV)
            # Discard if scraped value is implausibly low (< 25% of our estimate)
            if v and v >= fallback * 0.25:
                values[team] = v
                print(f"  [market_value] {team}: €{v}M (Transfermarkt live)")
            elif v:
                print(f"  [market_value] {team}: scraped €{v}M implausible, using fallback €{fallback}M")
            time.sleep(0.8)

    # Fallback for all remaining teams
    for team, mv in _FALLBACK_VALUES.items():
        if team not in values:
            values[team] = mv

    rows = [{"team": t, "market_value_eur_m": v} for t, v in sorted(values.items())]
    df = pd.DataFrame(rows)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)

    return values


def market_value_multiplier(
    team: str,
    opponent: str,
    mv_data: dict[str, float],
    power: float = 0.08,
) -> float:
    """
    Attack xG multiplier from squad market value ratio vs opponent.

    Spain (EUR 1060M) vs Cape Verde (EUR 40M): ratio=26.5 -> multiplier ~1.29
    Ecuador (EUR 180M) vs Germany (EUR 930M):  ratio=0.19 -> multiplier ~0.85

    power=0.08 gives gentle influence; largest realistic ratio (~30x) yields ~30%.
    """
    mv_a = mv_data.get(team,     _DEFAULT_MV)
    mv_b = mv_data.get(opponent, _DEFAULT_MV)
    ratio = mv_a / max(mv_b, 1.0)
    return float(math.exp(power * math.log(max(ratio, 0.01))))
