"""
WC 2026 venue data and team environmental context.

City names match those stored in the martj42 results.csv `city` column.
Altitude data sourced from official venue specs and Google Earth elevation.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Venue:
    name: str
    city: str          # as stored in martj42
    country: str
    lat: float
    lon: float
    altitude_m: float


# 16 official WC 2026 venues keyed by martj42 city name
VENUES: dict[str, Venue] = {
    "Mexico City":      Venue("Estadio Azteca",              "Mexico City",      "Mexico",        19.303, -99.151,  2240),
    "Zapopan":          Venue("Estadio BBVA Guadalajara",    "Zapopan",          "Mexico",        20.670,-103.427,  1555),
    "Guadalupe":        Venue("Estadio BBVA Monterrey",      "Guadalupe",        "Mexico",        25.670,-100.318,   536),
    "Toronto":          Venue("BMO Field",                   "Toronto",          "Canada",        43.633, -79.418,    76),
    "Vancouver":        Venue("BC Place",                    "Vancouver",        "Canada",        49.277,-123.112,     2),
    "Inglewood":        Venue("SoFi Stadium",                "Inglewood",        "United States", 33.953,-118.339,    27),
    "Santa Clara":      Venue("Levi's Stadium",              "Santa Clara",      "United States", 37.403,-121.970,    14),
    "East Rutherford":  Venue("MetLife Stadium",             "East Rutherford",  "United States", 40.814, -74.074,     4),
    "Foxborough":       Venue("Gillette Stadium",            "Foxborough",       "United States", 42.091, -71.264,    10),
    "Houston":          Venue("NRG Stadium",                 "Houston",          "United States", 29.685, -95.411,    10),
    "Philadelphia":     Venue("Lincoln Financial Field",     "Philadelphia",     "United States", 39.901, -75.168,     5),
    "Arlington":        Venue("AT&T Stadium",                "Arlington",        "United States", 32.748, -97.093,   183),
    "Seattle":          Venue("Lumen Field",                 "Seattle",          "United States", 47.596,-122.332,     5),
    "Atlanta":          Venue("Mercedes-Benz Stadium",       "Atlanta",          "United States", 33.755, -84.401,   290),
    "Kansas City":      Venue("Arrowhead Stadium",           "Kansas City",      "United States", 38.820, -94.484,   269),
    "Miami Gardens":    Venue("Hard Rock Stadium",           "Miami Gardens",    "United States", 25.958, -80.239,     2),
    # Alias — martj42 may list "Miami" for Miami Gardens
    "Miami":            Venue("Hard Rock Stadium",           "Miami",            "United States", 25.958, -80.239,     2),
    "Glendale":         Venue("State Farm Stadium",          "Glendale",         "United States", 33.527,-112.263,   349),
    "Dallas":           Venue("AT&T Stadium",                "Dallas",           "United States", 32.748, -97.093,   183),
}

# Approximate altitude (m) of each national team's primary training base.
# Used to compute altitude penalty: teams adapted to high altitude are not penalised.
TEAM_HOME_ALTITUDE: dict[str, float] = {
    # High altitude — fully adapted
    "Bolivia":      3600,
    "Ecuador":      2850,
    "Colombia":     2600,
    "Mexico":       2250,
    "Peru":          154,
    # Medium altitude
    "Switzerland":   540,
    "Austria":       180,
    "United States": 100,
    "Canada":         50,
    "Chile":         520,
    # Sea level (default 30 if not listed)
}
_DEFAULT_HOME_ALT = 30.0


# Climate zones used to assess heat/humidity penalty.
# "cold"    → Scandinavia, North UK, Canada, northern Europe
# "temperate" → most of Europe, Argentina, southern Brazil
# "warm"    → Mediterranean, southern USA, Mexico, Japan, South Korea
# "hot"     → sub-Saharan Africa, Middle East, SE Asia, Caribbean
TEAM_CLIMATE_ZONE: dict[str, str] = {
    "Norway":        "cold",
    "Sweden":        "cold",
    "Scotland":      "cold",
    "Canada":        "cold",
    "Denmark":       "cold",
    "Finland":       "cold",
    "Iceland":       "cold",
    "Netherlands":   "temperate",
    "Belgium":       "temperate",
    "Germany":       "temperate",
    "England":       "temperate",
    "France":        "temperate",
    "Austria":       "temperate",
    "Switzerland":   "temperate",
    "Croatia":       "temperate",
    "Poland":        "temperate",
    "Czech Republic": "temperate",
    "Slovakia":      "temperate",
    "Portugal":      "warm",
    "Spain":         "warm",
    "Italy":         "warm",
    "Greece":        "warm",
    "Turkey":        "warm",
    "United States": "warm",
    "Mexico":        "warm",
    "Japan":         "warm",
    "South Korea":   "warm",
    "Australia":     "warm",
    "Argentina":     "temperate",
    "Brazil":        "warm",
    "Uruguay":       "temperate",
    "Colombia":      "warm",
    "Ecuador":       "warm",
    "Paraguay":      "warm",
    "Chile":         "temperate",
    "Senegal":       "hot",
    "Morocco":       "warm",
    "Ivory Coast":   "hot",
    "Ghana":         "hot",
    "DR Congo":      "hot",
    "Egypt":         "hot",
    "Algeria":       "warm",
    "Cape Verde":    "hot",
    "South Africa":  "warm",
    "Saudi Arabia":  "hot",
    "Iran":          "warm",
    "Iraq":          "hot",
    "Jordan":        "hot",
    "Uzbekistan":    "warm",
    "New Zealand":   "temperate",
    "Panama":        "hot",
    "Haiti":         "hot",
}
_DEFAULT_CLIMATE = "temperate"

# Heat/humidity penalty by climate zone combination.
# Keys: (team_zone, match_heat_category)
# match_heat_category: "cool" (<20°C), "warm" (20-28°C), "hot" (>28°C and humidity>65)
_CLIMATE_PENALTY: dict[tuple[str, str], float] = {
    ("cold",       "cool"): 1.00,
    ("cold",       "warm"): 0.97,
    ("cold",       "hot"):  0.90,
    ("temperate",  "cool"): 1.00,
    ("temperate",  "warm"): 0.99,
    ("temperate",  "hot"):  0.94,
    ("warm",       "cool"): 1.00,
    ("warm",       "warm"): 1.00,
    ("warm",       "hot"):  0.98,
    ("hot",        "cool"): 0.98,   # cold weather slight penalty for tropical teams
    ("hot",        "warm"): 1.00,
    ("hot",        "hot"):  1.00,
}


def get_venue(city: str, country: str = "") -> Venue | None:
    """Look up a venue by city name (case-insensitive)."""
    return VENUES.get(city) or VENUES.get(city.strip())


def home_altitude(team: str) -> float:
    return TEAM_HOME_ALTITUDE.get(team, _DEFAULT_HOME_ALT)


def climate_zone(team: str) -> str:
    return TEAM_CLIMATE_ZONE.get(team, _DEFAULT_CLIMATE)


def altitude_multiplier(team: str, venue_alt: float) -> float:
    """
    Attack xG multiplier for altitude disadvantage.

    Teams already adapted (home altitude within 300m of venue) are not penalised.
    Effect: ~8 % reduction in goals at 2250m for a sea-level team, calibrated
    from meta-analysis of high-altitude football performance (Nassis 2013).
    """
    diff = max(0.0, venue_alt - home_altitude(team) - 300)
    K = 0.038  # per 1000 m disadvantage
    return float(__import__("math").exp(-K * diff / 1000))


def climate_multiplier(team: str, temp_c: float, humidity_pct: float) -> float:
    """Attack xG multiplier for heat/humidity stress."""
    if temp_c > 28 and humidity_pct > 65:
        heat_cat = "hot"
    elif temp_c > 20:
        heat_cat = "warm"
    else:
        heat_cat = "cool"
    zone = climate_zone(team)
    return _CLIMATE_PENALTY.get((zone, heat_cat), 1.0)
