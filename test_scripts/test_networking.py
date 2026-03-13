import socket
from unittest.mock import patch

from openforge.utils.networking import (
    format_uri_host,
    get_host_ip,
    normalize_address_port,
    normalize_ip_address,
)


def test_normalize_ip_address_keeps_ipv4() -> None:
    assert normalize_ip_address("127.0.0.1") == "127.0.0.1"


def test_normalize_ip_address_unwraps_ipv4_mapped_ipv6() -> None:
    assert normalize_ip_address("::ffff:127.0.0.1") == "127.0.0.1"


def test_normalize_ip_address_strips_brackets() -> None:
    assert normalize_ip_address("[2001:db8::1]") == "2001:db8::1"


def test_format_uri_host_brackets_ipv6_only() -> None:
    assert format_uri_host("127.0.0.1") == "127.0.0.1"
    assert format_uri_host("2001:db8::1") == "[2001:db8::1]"


def test_normalize_address_port_formats_url_authority() -> None:
    assert normalize_address_port("127.0.0.1", 8080) == "127.0.0.1:8080"
    assert normalize_address_port("2001:db8::1", 8080) == "[2001:db8::1]:8080"


def test_get_host_ip_prefers_non_loopback_ipv4() -> None:
    with (
        patch("openforge.utils.networking.socket.gethostname", return_value="test-host"),
        patch(
            "openforge.utils.networking.socket.getaddrinfo",
            return_value=[
                (socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("2001:db8::1", 0, 0, 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.12", 0)),
            ],
        ),
    ):
        assert get_host_ip() == "10.0.0.12"


def test_get_host_ip_falls_back_to_loopback_ipv4() -> None:
    with (
        patch("openforge.utils.networking.socket.gethostname", return_value="test-host"),
        patch(
            "openforge.utils.networking.socket.getaddrinfo",
            return_value=[
                (socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("::1", 0, 0, 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0)),
            ],
        ),
    ):
        assert get_host_ip() == "127.0.0.1"


def test_get_host_ip_falls_back_to_ipv6_when_no_ipv4() -> None:
    with (
        patch("openforge.utils.networking.socket.gethostname", return_value="test-host"),
        patch(
            "openforge.utils.networking.socket.getaddrinfo",
            return_value=[
                (socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("2001:db8::1", 0, 0, 0)),
            ],
        ),
    ):
        assert get_host_ip() == "2001:db8::1"


def test_get_host_ip_raises_when_no_addresses_found() -> None:
    with (
        patch("openforge.utils.networking.socket.gethostname", return_value="test-host"),
        patch("openforge.utils.networking.socket.getaddrinfo", return_value=[]),
    ):
        try:
            get_host_ip()
        except RuntimeError as exc:
            assert str(exc) == "could not determine host IP address"
        else:
            raise AssertionError("expected get_host_ip() to raise RuntimeError")


def main() -> int:
    tests = [
        test_normalize_ip_address_keeps_ipv4,
        test_normalize_ip_address_unwraps_ipv4_mapped_ipv6,
        test_normalize_ip_address_strips_brackets,
        test_format_uri_host_brackets_ipv6_only,
        test_normalize_address_port_formats_url_authority,
        test_get_host_ip_prefers_non_loopback_ipv4,
        test_get_host_ip_falls_back_to_loopback_ipv4,
        test_get_host_ip_falls_back_to_ipv6_when_no_ipv4,
        test_get_host_ip_raises_when_no_addresses_found,
    ]

    for test in tests:
        test()

    print(f"SUCCESS ran {len(tests)} networking tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
