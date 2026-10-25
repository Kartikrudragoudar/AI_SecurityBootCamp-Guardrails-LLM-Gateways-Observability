import os
from typing import Any, Optional
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage


class Mistral:
    def __init__(
        self,
        model: str = "ministral-8b-latest",
        api_key: Optional[str] = None,
        base_url: str = "https://api.mistral.ai/v1",
    ):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY", "")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
        )
        self.model = model

    def add_user_message(self, messages: list, message: Any):
        if isinstance(message, dict):
            messages.append(message)
        elif isinstance(message, list):
            messages.extend(message)
        elif hasattr(message, "content"):
            messages.append({"role": "user", "content": message.content})
        else:
            messages.append({"role": "user", "content": str(message)})

    def add_assistant_message(self, messages: list, message: Any):
        if isinstance(message, dict):
            messages.append(message)
            return

        msg_dict: dict[str, Any] = {"role": "assistant"}
        if hasattr(message, "content") and message.content:
            msg_dict["content"] = message.content
        else:
            msg_dict["content"] = ""

        if hasattr(message, "tool_calls") and message.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        messages.append(msg_dict)

    def text_from_message(self, message: Any) -> str:
        if isinstance(message, str):
            return message
        if hasattr(message, "content") and message.content:
            return message.content
        if isinstance(message, dict):
            return message.get("content", "") or ""
        return ""

    def chat(
        self,
        messages: list,
        system: Optional[str] = None,
        temperature: float = 0.7,
        tools: Optional[list] = None,
    ) -> ChatCompletionMessage:
        formatted_messages = []
        if system:
            formatted_messages.append({"role": "system", "content": system})
        formatted_messages.extend(messages)

        params: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
        }

        if tools:
            params["tools"] = tools

        response = self.client.chat.completions.create(**params)
        return response.choices[0].message

