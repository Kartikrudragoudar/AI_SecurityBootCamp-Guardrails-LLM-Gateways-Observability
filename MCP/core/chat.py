from typing import Any
from core.mistral import Mistral
from mcp_client import MCPClient
from core.tools import ToolManager


class Chat:
    def __init__(self, mistral_service: Mistral, clients: dict[str, MCPClient]):
        self.mistral_service: Mistral = mistral_service
        self.clients: dict[str, MCPClient] = clients
        self.messages: list[dict[str, Any]] = []

    async def _process_query(self, query: str):
        self.messages.append({"role": "user", "content": query})

    async def run(
        self,
        query: str,
    ) -> str:
        final_text_response = ""

        await self._process_query(query)

        while True:
            tools = await ToolManager.get_all_tools(self.clients)
            response = self.mistral_service.chat(
                messages=self.messages,
                tools=tools if tools else None,
            )

            self.mistral_service.add_assistant_message(self.messages, response)

            if getattr(response, "tool_calls", None):
                text = self.mistral_service.text_from_message(response)
                if text:
                    print(text)

                tool_messages = await ToolManager.execute_tool_requests(
                    self.clients, response
                )
                self.messages.extend(tool_messages)
            else:
                final_text_response = self.mistral_service.text_from_message(
                    response
                )
                break

        return final_text_response
