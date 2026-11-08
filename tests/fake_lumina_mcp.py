"""Minimal stand-in for Lumina's MCP stdio CLI (used by tests/test_wiki.py)."""

from __future__ import annotations

import json
import os
import sys
import time

args = sys.argv[1:]
vault = args[args.index("--vault") + 1] if "--vault" in args else ""
crash_file = os.environ.get("FAKE_LUMINA_CRASH_ONCE")

PAGES = {
    "wiki/concepts/pricing.md": {"title": "Pricing model", "body": "Tiered pricing: Basic $10, Pro $25."},
    "wiki/entities/acme.md": {"title": "Acme Corp", "body": "Acme is our pilot client in Osaka."},
}


def reply(rid, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": rid}
    if error:
        msg["error"] = error
    else:
        msg["result"] = result
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def text(obj, is_error=False):
    out = {"content": [{"type": "text", "text": obj if isinstance(obj, str) else json.dumps(obj)}]}
    if is_error:
        out["isError"] = True
    return out


for line in sys.stdin:
    msg = json.loads(line)
    rid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
    if method == "initialize":
        reply(rid, {"protocolVersion": params.get("protocolVersion"), "capabilities": {"tools": {}}, "serverInfo": {"name": "lumina", "version": "0"}})
    elif method == "tools/call":
        name, a = params.get("name"), params.get("arguments") or {}
        if crash_file and os.path.exists(crash_file):
            os.remove(crash_file)
            sys.exit(1)
        if name == "search_wiki":
            if a.get("query") == "slow":
                time.sleep(5)
            hits = [
                {"title": "Raw note", "filename": "raw/notes.md", "excerpt": "raw excerpt"},
                {"title": "Pricing model", "filename": "wiki/concepts/pricing.md", "excerpt": f"pricing in {os.path.basename(vault)}"},
                {"title": "Acme Corp", "filename": "wiki/entities/acme.md", "excerpt": "acme"},
            ]
            reply(rid, text(hits[: a.get("limit", 8)]))
        elif name == "get_article":
            page = PAGES.get(a.get("filename"))
            if page is None:
                reply(rid, text(f"{a.get('filename')} not found", is_error=True))
            else:
                reply(rid, text({"filename": a["filename"], "frontmatter": {"title": page["title"]}, "body": page["body"]}))
        elif name == "list_articles":
            reply(rid, text({"total": 2, "articles": [{"title": p["title"], "filename": f} for f, p in PAGES.items()]}))
        else:
            reply(rid, error={"code": -32602, "message": f"Tool {name} not found"})
    elif rid is not None:
        reply(rid, {})
