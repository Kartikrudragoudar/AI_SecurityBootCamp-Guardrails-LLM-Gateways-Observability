import os
from getpass import getpass
from dotenv import load_dotenv
from meshapi import MeshAPI
from meshapi import ChatCompletionParams, ChatMessage
load_dotenv()

MESHAPI_TOKEN = os.getenv("MESH_API_KEY") or getpass("MeshAPI token(rsk_...)")

client = MeshAPI(
    base_url=os.getenv("MESH_BASE_URL", "https://api.meshapi.ai"),
    token=MESHAPI_TOKEN
)

ALL_MODELS = {m.id: m for m in client.models.list()}
print(f"{len(ALL_MODELS)} models available through this key")


ALL_MODELS_BY_NAME = {m.name: m for m in ALL_MODELS.values()}
ALL_MODELS_BY_NAME[list(ALL_MODELS_BY_NAME.keys())[0]]

resp = client.chat.completions.create(
    ChatCompletionParams(
        model="ministral-8b-latest",
        messages=[
            ChatMessage(
                role="user",
                content="In One Sentence, What is an AI gateway"
            )
        ],
        max_tokens=60
    )
)

print(resp.choices[0].message.content)