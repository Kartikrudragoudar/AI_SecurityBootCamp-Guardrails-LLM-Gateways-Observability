from tensorzero import TensorZeroGateway

TEXT_TO_SUMMARIZE = """
Artificial intelligence has experienced remarkable growth over the past decade,
fundamentally changing how software systems are built and deployed. Large Language
Models (LLMs) like GPT-4 and Llama have demonstrated capabilities ranging from
writing code to complex reasoning. However, managing these models in production
presents new challenges: teams need to monitor performance, control costs, run
experiments, and continuously improve their AI features. LLM gateways have emerged
as a solution, acting as an intelligent middleware layer between applications and
the underlying model providers.
"""

with TensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:

    print("Making inference via TensorZero native client...")

    result = client.inference(
        function_name = "summarize",
        input = {
            "messages":[
                {
                    "role" : "user",
                    "content" : TEXT_TO_SUMMARIZE
                }
            ]
        }
    )

    print(f" Type: {type(result).__name__}")
    print(f"inference_id:{result.inference_id}")
    print(f"episode_id:{result.episode_id}")
    print(f"variant_name:{result.variant_name}")


    summary_text = result.content[0].text
    print(summary_text)