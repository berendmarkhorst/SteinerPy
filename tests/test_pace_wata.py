import pytest

from benchmarks.pace_wata import parse_solution


def test_wata_certificate_parser():
    assert parse_solution("VALUE 7\n1 2\n2 3\n") == (7, [(1, 2), (2, 3)])
    for text in ("", "7\n1 2", "VALUE 7\n1 2 3", "VALUE 7\n1 x"):
        with pytest.raises(ValueError):
            parse_solution(text)
