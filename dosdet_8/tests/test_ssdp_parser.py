from dosdet.features.ssdp_parser import parse_ssdp_payload

def test_parse_msearch():
    raw = b"M-SEARCH * HTTP/1.1\r\nMAN: \"ssdp:discover\"\r\nST: upnp:rootdevice\r\nUSER-AGENT: TestAgent/1.0\r\n\r\n"
    method, tokens = parse_ssdp_payload(raw)
    assert method == "M-SEARCH"
    assert tokens["MAN"].lower().startswith('"ssdp:discover"')
    assert tokens["ST"] == "upnp:rootdevice"
    assert "USER-AGENT" in tokens

def test_parse_200ok():
    raw = b"HTTP/1.1 200 OK\r\nST: upnp:rootdevice\r\n\r\n"
    method, tokens = parse_ssdp_payload(raw)
    assert method == "200-OK"
    assert tokens["ST"] == "upnp:rootdevice"
