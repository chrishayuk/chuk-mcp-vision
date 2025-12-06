"""Vision MCP Server - Generic computer vision primitives.

Provides atomic, composable vision tools that can be orchestrated by
chuk-ai-planner to solve complex vision tasks.
"""

import logging
import sys

from chuk_mcp_server import run

# Import all tools (they self-register via @tool decorator)

# Configure logging
logging.basicConfig(
    level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s", stream=sys.stderr
)
logger = logging.getLogger(__name__)


def main():
    """Run the Vision MCP server."""
    transport = "stdio"

    if len(sys.argv) > 1 and sys.argv[1] in ["http", "--http"]:
        transport = "http"
        logger.warning("Starting Chuk MCP Vision Server in HTTP mode")

    if transport == "stdio":
        logging.getLogger("chuk_mcp_server").setLevel(logging.ERROR)
        logging.getLogger("chuk_mcp_server.core").setLevel(logging.ERROR)
        logging.getLogger("chuk_mcp_server.stdio_transport").setLevel(logging.ERROR)

    run(transport=transport)


if __name__ == "__main__":
    main()
