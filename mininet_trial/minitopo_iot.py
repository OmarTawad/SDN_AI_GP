#!/usr/bin/env python3
from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel

# ------- Host inventory (short names; <=10 chars) -------
# ip must include mask (/24). mac must be lowercase 6-byte.
HOSTS = {
    "amazon_ech": {"ip": "192.168.1.239/24", "mac": "44:65:0d:56:cc:d3"},
    "belkin_wem": {"ip": "192.168.1.165/24", "mac": "ec:1a:59:83:28:11"},
    "dropcam":    {"ip": "192.168.1.106/24", "mac": "30:8c:fb:2f:e4:b2"},
    "hp_printer": {"ip": "192.168.1.236/24", "mac": "70:5a:0f:e4:9b:c0"},
    "ihome":      {"ip": "192.168.1.163/24", "mac": "74:c6:3b:29:d7:1d"},
    "lifx_smart": {"ip": "192.168.1.118/24", "mac": "d0:73:d5:01:83:08"},
    "nest_prote": {"ip": "192.168.1.168/24", "mac": "18:b4:30:25:be:e4"},
    "netatmo_we": {"ip": "192.168.1.112/24", "mac": "70:ee:50:03:b8:ac"},
    "pix_star_p": {"ip": "192.168.1.177/24", "mac": "e0:76:d0:33:bb:85"},
    "samsung_sm": {"ip": "192.168.1.248/24", "mac": "00:16:6c:ab:6b:88"},
    "smart_thin": {"ip": "192.168.1.196/24", "mac": "d0:52:a8:00:67:5e"},
    "tp_link_da": {"ip": "192.168.1.108/24", "mac": "f4:f2:6d:93:51:f1"},
    "tp_link_sm": {"ip": "192.168.1.227/24", "mac": "50:c7:bf:00:56:39"},
    "tplink_rou": {"ip": "192.168.1.1/24",   "mac": "14:cc:20:51:33:ea"},  # "gateway" host
    "triby_spea": {"ip": "192.168.1.120/24", "mac": "18:b7:9e:02:20:44"},
    "unknown_00": {"ip": "192.168.1.129/24", "mac": "00:17:88:2b:9a:25"},
    "unknown_2c": {"ip": "192.168.1.205/24", "mac": "2c:27:d7:3b:e1:05"},
    "unknown_7c": {"ip": "192.168.1.230/24", "mac": "7c:70:bc:5d:5e:dc"},
    "unknown_08": {"ip": "192.168.1.175/24", "mac": "08:2e:5f:25:88:e5"},
    "unknown_84": {"ip": "192.168.1.219/24", "mac": "84:f3:eb:52:42:db"},
    "unknown_88": {"ip": "192.168.1.221/24", "mac": "88:4a:ea:31:66:9d"},
    "unknown_b4": {"ip": "192.168.1.245/24", "mac": "b4:75:0e:ec:e5:a9"},
    "unknown_e0": {"ip": "192.168.1.216/24", "mac": "e0:76:d0:3f:00:ae"},
    "unknown_ec": {"ip": "192.168.1.169/24", "mac": "ec:1a:59:7a:02:c5"},
    "unknown_f2": {"ip": "192.168.1.192/24", "mac": "f4:f5:d8:d4:eb:12"},
    "unknown_f4": {"ip": "192.168.1.119/24", "mac": "f4:f5:d8:8f:0a:3c"},
}

# ------- Link shaping defaults (all links) -------
LINK_KW = dict(
    cls=TCLink,
    bw=1,                 # 1 Mbit/s
    delay='2ms',          # tiny propagation (keeps TCP sane)
    max_queue_size=50,    # small buffer -> faster loss under burst
    use_htb=True
)

def build_net():
    # Controller: Ryu listening on 127.0.0.1:6633
    net = Mininet(
        controller=None,                # we'll add RemoteController explicitly
        switch=OVSSwitch,
        link=TCLink,
        autoStaticArp=True,             # ARP convenience
        waitConnected=True
    )

    c0 = net.addController('c0', controller=RemoteController, ip='127.0.0.1', port=6633)
    s1 = net.addSwitch('s1', protocols='OpenFlow13')

    name_to_host = {}
    for name, attrs in HOSTS.items():
        h = net.addHost(
            name, ip=attrs['ip'], mac=attrs['mac']
        )
        name_to_host[name] = h
        net.addLink(h, s1, **LINK_KW)

    net.start()

    # Optional: add default route via "tplink_rou" if you want hosts to try off-subnet
    # for name, host in name_to_host.items():
    #     if name != 'tplink_rou':
    #         host.cmd('ip route replace default via 192.168.1.1 dev {}-eth0'.format(name))

    print("\n*** Hosts (name/IP/MAC):")
    for name in sorted(name_to_host.keys()):
        h = name_to_host[name]
        # fetch effective IP/MAC from the host
        ip = h.IP()
        mac = h.MAC()
        print(f"{name:11s} {ip:15s} {mac}")

    print("\nTry: pingall, or `xterm <host>` and run tcpdump while replay runs.")
    return net

if __name__ == '__main__':
    setLogLevel('info')
    net = build_net()
    try:
        CLI(net)
    finally:
        net.stop()
