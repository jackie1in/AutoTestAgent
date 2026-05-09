import os

DEFAULT_START_URL = os.getenv("MAPPING_URL", "https://the-internet.herokuapp.com/login")
DEFAULT_EXPECTED_END_URL = (
    "https://the-internet.herokuapp.com/secure"
)

_ALLOWED_KNOWLEDGE_PROFILES = {"conservative", "balanced", "aggressive"}


def _normalize_knowledge_profile(value: str) -> str:
    profile = (value or "").strip().lower()
    if profile in _ALLOWED_KNOWLEDGE_PROFILES:
        return profile
    return "balanced"


def _resolve_app_name() -> str:
    return (os.getenv("MAPPING_APP_NAME") or "").strip()


def _resolve_release_id() -> str:
    return (os.getenv("MAPPING_RELEASE_ID") or "").strip()
