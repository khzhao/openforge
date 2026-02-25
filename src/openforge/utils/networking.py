# Copyright 2026 openforge

import socket


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
