"""Fail-closed checks for the optional loopback-only local demonstration mode."""

from urllib.parse import urlsplit


LOCAL_DEMO_ENV = "FL_LOCAL_DEMO_MODE"
_LOOPBACK_ADDRESSES = {"127.0.0.1", "::1", "localhost"}
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def local_demo_mode_enabled(environment):
    """Enable local demo only for the exact explicit opt-in value."""
    return environment.get(LOCAL_DEMO_ENV, "0") == "1"


def is_local_demo_request(server_address, headers):
    """Allow automatic demo sign-in only on a direct loopback connection."""
    if (server_address or "").strip().lower() not in _LOOPBACK_ADDRESSES:
        return False

    lowered_headers = {str(key).lower(): value for key, value in headers.items()}
    if "x-forwarded-host" in lowered_headers or "x-forwarded-for" in lowered_headers:
        return False

    host_header = lowered_headers.get("host")
    if not isinstance(host_header, str) or not host_header.strip():
        return False
    authority = host_header.strip()
    if "," in authority:
        return False
    try:
        parsed = urlsplit(f"//{authority}")
        hostname = parsed.hostname
        parsed.port
    except ValueError:
        return False
    return (
        hostname in _LOOPBACK_HOSTS
        and parsed.username is None
        and parsed.password is None
    )
