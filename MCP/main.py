import asyncio
import sys
import os
from dotenv import load_dotenv
from contextlib import AsyncExitStack

from mcp_client import MCPClient
from core.mistral import Mistral

from core.cli_chat import CliChat
from core.cli import CliApp

load_dotenv()

# Mistral Config
mistral_model = os.getenv("MISTRAL_MODEL", "ministral-8b-latest")
mistral_api_key = os.getenv("MISTRAL_API_KEY", "")


assert mistral_model, "Error: MISTRAL_MODEL cannot be empty. Update .env"
assert mistral_api_key, (
    "Error: MISTRAL_API_KEY cannot be empty. Update .env"
)


async def main():
    mistral_service = Mistral(model=mistral_model, api_key=mistral_api_key)

    server_scripts = sys.argv[1:]
    clients = {}

    python_executable = sys.executable

    command, args = (
        ("uv", ["run", "mcp_server.py"])
        if os.getenv("USE_UV", "0") == "1"
        else (python_executable, ["mcp_server.py"])
    )

    async with AsyncExitStack() as stack:
        doc_client = await stack.enter_async_context(
            MCPClient(command=command, args=args)
        )
        clients["doc_client"] = doc_client

        for i, server_script in enumerate(server_scripts):
            client_id = f"client_{i}_{server_script}"
            client = await stack.enter_async_context(
                MCPClient(command=python_executable, args=[server_script])
            )
            clients[client_id] = client

        chat = CliChat(
            doc_client=doc_client,
            clients=clients,
            mistral_service=mistral_service,
        )

        cli = CliApp(chat)
        await cli.initialize()
        await cli.run()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
