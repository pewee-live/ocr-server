"""End-to-end MCP smoke test over stdio.

Spawns the server as a subprocess, lists tools, and calls recognize_text on the
English sample image. Usage: python tests/test_mcp_client.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "sample_data" / "en.png"


async def amain() -> int:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "ocr_mcp"],
        cwd=str(ROOT),
        env={"MCP_TRANSPORT": "stdio", "PYTHONPATH": str(ROOT)},
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("tools:", [t.name for t in tools.tools])

            langs = await session.call_tool("list_supported_languages", {})
            print("langs:", langs.content)

            result = await session.call_tool(
                "recognize_text",
                {"image": str(SAMPLE), "language": "en", "detail": True},
            )
            for block in result.content:
                if hasattr(block, "text"):
                    print("recognize_text ->", block.text)
                    if "paddleocr" not in block.text.lower():
                        return 1
            return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    sys.exit(main())
