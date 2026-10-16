"""Optional phone push when orders need your approval, through ntfy (https://ntfy.sh).
Install the ntfy app, subscribe to a long random topic name, and put it in the
config as execution.ntfy_topic. Messages carry the order list only, never keys
or balances. Off unless a topic is set; a failed push never stops the scanner."""
from __future__ import annotations

import urllib.request


def push(cfg, title, body, click=None):
    ex = cfg.get("execution", {})
    topic = ex.get("ntfy_topic")
    if not topic:
        return False
    req = urllib.request.Request(f"{ex.get('ntfy_server', 'https://ntfy.sh').rstrip('/')}/{topic}", data=body.encode(),
                                 headers={"Title": title, "Priority": "high", "Tags": "moneybag"} | ({"Click": click} if click else {}))
    try:
        urllib.request.urlopen(req, timeout=10).close()
        return True
    except OSError:
        return False
