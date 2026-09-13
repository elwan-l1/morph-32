"""Three-plane conversion profiles."""

PROFILES = ("morph32-3s", "morph32-c", "independent-3s", "affine-3bit")


def normalize_profile(profile):
    """Validate a public profile identifier."""
    if profile not in PROFILES:
        raise ValueError(f"Unsupported profile: {profile}")
    return profile


def recipe(profile, layer, projection):
    profile = normalize_profile(profile)
    if not 0 <= layer < 64:
        raise ValueError("Unsupported profile or layer")
    if projection not in ("gate_proj", "up_proj", "down_proj"):
        raise ValueError("Unsupported MLP projection")
    spec = dict(seeds=3, planes=3, decay=0.5)
    if profile == "independent-3s":
        # Same fitter, scale, layout and kernels; the recurrence is the only removal.
        spec["planes"] = 3
    if profile == "affine-3bit":
        return dict(bits=3, group_size=64)
    if profile == "morph32-c" and 8 <= layer <= 55 and projection != "down_proj":
        spec["partial"] = True
    return spec


def seed_search(profile):
    """Additional refinement used by current public profiles, after the base fit."""
    profile = normalize_profile(profile)
    if profile in ("morph32-3s", "morph32-c"):
        return dict(sweeps=2, restarts=3, seed=2026)
    return None
