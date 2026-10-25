import json
from typing import Optional, Any
from mcp.types import CallToolResult, TextContent
from mcp_client import MCPClient


class ToolManager:
    @classmethod
    async def get_all_tools(cls, clients: dict[str, MCPClient]) -> list[dict]:
        """Gets all tools from the provided clients formatted for Mistral / OpenAI."""
        tools = []
        for client in clients.values():
            tool_models = await client.list_tools()
            tools += [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": getattr(t, "input_schema", None)
                        or getattr(t, "inputSchema", None)
                        or {},
                    },
                }
                for t in tool_models
            ]
        return tools

    @classmethod
    async def _find_client_with_tool(
        cls, clients: list[MCPClient], tool_name: str
    ) -> Optional[MCPClient]:
        """Finds the first client that has the specified tool."""
        for client in clients:
            tools = await client.list_tools()
            tool = next((t for t in tools if t.name == tool_name), None)
            if tool:
                return client
        return None

    @classmethod
    async def execute_tool_requests(
        cls, clients: dict[str, MCPClient], message: Any
    ) -> list[dict]:
        """Executes a list of tool requests against the provided clients."""
        tool_calls = getattr(message, "tool_calls", None) or []
        tool_messages: list[dict] = []

        for tool_call in tool_calls:
            tool_use_id = tool_call.id
            tool_name = tool_call.function.name

            try:
                if isinstance(tool_call.function.arguments, str):
                    tool_input = json.loads(tool_call.function.arguments)
                elif isinstance(tool_call.function.arguments, dict):
                    tool_input = tool_call.function.arguments
                else:
                    tool_input = {}
            except Exception:
                tool_input = {}

            client = await cls._find_client_with_tool(
                list(clients.values()), tool_name
            )

            if not client:
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "name": tool_name,
                    "content": json.dumps({"error": f"Could not find tool '{tool_name}'"}),
                })
                continue

            try:
                tool_output: CallToolResult | None = await client.call_tool(
                    tool_name, tool_input
                )
                items = []
                if tool_output:
                    items = tool_output.content
                content_list = [
                    item.text for item in items if isinstance(item, TextContent)
                ]
                content_str = "\n".join(content_list) if content_list else ""
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "name": tool_name,
                    "content": content_str,
                })
            except Exception as e:
                error_message = f"Error executing tool '{tool_name}': {e}"
                print(error_message)
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "name": tool_name,
                    "content": json.dumps({"error": error_message}),
                })

        return tool_messages
