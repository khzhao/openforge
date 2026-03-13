# Copyright 2026 openforge

import ipaddress
import socket


def normalize_ip_address(addr: str) -> str:
    """Normalize an IP address, preferring canonical IPv4 when available."""
    ip = ipaddress.ip_address(addr.strip().strip("[]"))
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return str(ip.ipv4_mapped)
    return ip.compressed


def format_uri_host(addr: str) -> str:
    """Format an IP address for use in a URL host component."""
    normalized = normalize_ip_address(addr)
    ip = ipaddress.ip_address(normalized)
    if isinstance(ip, ipaddress.IPv6Address):
        return f"[{normalized}]"
    return normalized


def normalize_address_port(addr: str, port: int) -> str:
    """Normalize an IP address and port for use in a URL authority."""
    return f"{format_uri_host(addr)}:{port}"


def get_host_ip() -> str:
    """Return the current host's IP, preferring a non-loopback IPv4 address."""
    hostname = socket.gethostname()
    addrinfos = socket.getaddrinfo(
        hostname,
        None,
        socket.AF_UNSPEC,
        socket.SOCK_STREAM,
    )

    ipv4_fallback: str | None = None
    ipv6_fallback: str | None = None
    for family, _socktype, _proto, _canonname, sockaddr in addrinfos:
        normalized = normalize_ip_address(sockaddr[0])
        ip = ipaddress.ip_address(normalized)
        if family == socket.AF_INET:
            if not ip.is_loopback:
                return normalized
            if ipv4_fallback is None:
                ipv4_fallback = normalized
            continue
        if family == socket.AF_INET6:
            if not ip.is_loopback and ipv6_fallback is None:
                ipv6_fallback = normalized
            elif ipv6_fallback is None:
                ipv6_fallback = normalized

    if ipv4_fallback is not None:
        return ipv4_fallback
    if ipv6_fallback is not None:
        return ipv6_fallback
    raise RuntimeError("could not determine host IP address")


def is_port_free(port: int) -> bool:
    """Check if a port is free."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            return True
        except OSError:
            return False


def get_free_port(start: int = 10000, block_size: int = 1) -> int:
    """Get the first port whose next block_size ports are free."""
    while not all(is_port_free(port) for port in range(start, start + block_size)):
        start += 1
    return start
