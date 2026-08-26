from orbit.tools import get_tool


def test_fetch_rejects_non_http():
    fetch = get_tool("fetch_url")
    r = fetch.execute(url="file:///etc/passwd")
    assert "only http" in r
