"""
WC 2026 group definitions and team name mappings.

Groups confirmed after draw (Dec 5 2025) + playoff results (Mar 2026).
UEFA playoff winners: Bosnia and Herzegovina, Czechia, Sweden, Türkiye
Intercontinental playoff winners: DR Congo (K), Iraq (I)
"""

# Groups use DISPLAY names; MODEL_NAME maps these to the martj42 dataset naming
GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Maps WC 2026 display name → martj42 dataset name.
# Only entries that differ are listed; identity mapping is the default.
DISPLAY_TO_MODEL: dict[str, str] = {
    "Czechia":                  "Czech Republic",
    "Türkiye":                  "Turkey",
    "Bosnia and Herzegovina":   "Bosnia-Herzegovina",
    "Curaçao":                  "Curacao",
    "Ivory Coast":              "Ivory Coast",   # dataset uses this name
    "South Korea":              "South Korea",
    "United States":            "United States",
    "DR Congo":                 "DR Congo",
    "Cape Verde":               "Cape Verde",
    "Saudi Arabia":             "Saudi Arabia",
    "New Zealand":              "New Zealand",
}

# Flat list of all 48 WC teams (display names)
WC_TEAMS: list[str] = [t for teams in GROUPS.values() for t in teams]


def to_model_name(display: str) -> str:
    """Convert WC display name to the name used in the martj42 dataset."""
    return DISPLAY_TO_MODEL.get(display, display)
