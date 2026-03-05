# Copyright 2026 openforge

import ipaddress
import socket


def format_v6_uri(addr: str) -> str:
    """Normalize an IP address to IPv6 URI form '[addr]'."""
    ip = ipaddress.ip_address(addr.strip())
    if isinstance(ip, ipaddress.IPv4Address):
        ip = ipaddress.IPv6Address("::ffff:" + str(ip))
    return f"[{ip.compressed}]"


def normalize_to_ipv6_address_port(addr: str, port: int) -> str:
    """Normalize an IP address and port to IPv6 format '[addr]:port'."""
    return f"{format_v6_uri(addr)}:{port}"


def is_port_free(port: int) -> bool:
    """Check if a port is free."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("localhost", port))
            return True
        except OSError:
            return False


def get_free_port(start: int = 10000, block_size: int = 1) -> int:
    """Get the first port whose next block_size ports are free."""
    while not all(is_port_free(port) for port in range(start, start + block_size)):
        start += 1
    return start
