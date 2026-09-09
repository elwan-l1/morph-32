"""The two measured recipes; no automatic quality-dependent selection."""

PROFILES = ("morph32-3s", "morph32-c")
PROFILE_ALIASES = {"three-seed": "morph32-3s", "compact": "morph32-c"}


def normalize_profile(profile):
    """Resolve previous release names to the public profile identifiers."""
    profile = PROFILE_ALIASES.get(profile, profile)
    if profile not in PROFILES:
        raise ValueError(f"Unsupported profile: {profile}")
    return profile


def recipe(profile, layer, projection):
    profile = normalize_profile(profile)
    if not 0 <= layer < 64:
        raise ValueError("Unsupported profile or layer")
    if projection not in ("gate_proj", "up_proj", "down_proj"):
        raise ValueError("Unsupported MLP projection")
    spec = dict(seeds=3, planes=5, rotations=[5, 13, 21], decay=0.5, lag=2, tail=0.5)
    if profile == "morph32-c" and 8 <= layer <= 55 and projection != "down_proj":
        spec["partial"] = True
    return spec
