"""
Minimal MCP server launcher with 3 modes:

1) stdio (default): blocks, waiting for an MCP client over stdin/stdout.
2) --list-tools: prints available tool specs and exits.
3) --run-tool <name> --input '<json>': executes a tool once and prints JSON.

Examples:
  python -m src.mcp.server --list-tools
  python -m src.mcp.server --run-tool validate_labels --input '{"path":"outputs/run_simple/labels/bitcoin*.labels.jsonl"}'
  python -m src.mcp.server --run-tool push_labels --input '{"out_dir":"outputs/run_simple"}'
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

from . import tools


def _stdio_mode() -> None:
    # In stdio mode we simply block waiting for an MCP client.
    # We print a small banner so it’s not “silent”.
    banner = (
        "MCP server (stdio mode): waiting for a client on stdin/stdout.\n"
        "Tip: For a quick local test, use:\n"
        "  python -m src.mcp.server --list-tools\n"
        "  python -m src.mcp.server --run-tool <name> --input '<json>'\n"
        "Press Ctrl+C to exit.\n"
    )
    sys.stderr.write(banner)
    sys.stderr.flush()
    # Real MCP hosts speak JSON-RPC here. We just block until the host connects.
    for _ in sys.stdin:
        # No-op: in a real host, JSON-RPC requests would be handled here.
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP server launcher")
    parser.add_argument("--list-tools", action="store_true", help="List available tools and exit")
    parser.add_argument("--run-tool", type=str, help="Run a single tool by name and print the result")
    parser.add_argument("--input", type=str, default="{}", help="JSON input for --run-tool")
    args = parser.parse_args()

    if args.list_tools:
        spec = tools.list_tools()
        print(json.dumps(spec, indent=2))
        return

    if args.run_tool:
        try:
            payload: Dict[str, Any] = json.loads(args.input) if args.input else {}
        except Exception as e:
            print(json.dumps({"ok": False, "error": f"Invalid --input JSON: {e}"}))
            return
        result = tools.run_tool(args.run_tool, payload)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    _stdio_mode()


if __name__ == "__main__":
    main()
