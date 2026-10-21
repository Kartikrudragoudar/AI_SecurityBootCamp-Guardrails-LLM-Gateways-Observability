from tensorzero import AsyncTensorZeroGateway
import asyncio

# with TensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:

#     r1 = client.inference(
#         function_name="chat",
#         input={
#             "messages":[
#                 {"role":"user", "content":"Name 3 benefits of using an LLM gateway"}
#             ]
#         }
#     )

#     episode_id = r1.episode_id
#     print(episode_id)
#     print(f"Turn 1 [inference_id={r1.inference_id}]:")
#     print(r1.content[1].text, "...")
#     print()


#     r2 = client.inference(
#         function_name="chat",
#         input={
#             "messages":[
#                 {"role":"user", "content":"Which of these is most important for startups?"}
#             ]
#         },
#         episode_id=episode_id
#     )

#     print(f"Turn 2 [inference_id={r2.inference_id}]:")
#     print(r2.content[1].text, "...")
#     print()

#     print(f"Both turns share episode ID: {episode_id}")
#     assert r1.episode_id == r2.episode_id == episode_id, "Episode IDs should match!"
#     print("Assertion passed: episode IDs match!")


async def run_parallel_inferences():

    async with await AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:

        print("Launching 2 inferences in parallel...")

        task1 = client.inference(
            function_name="summarize",
            input={"messages": [{"role": "user",
                                 "content": "Who was Archimedes and what were his contributions to science?"}]}
        )
        task2 = client.inference(
            function_name="summarize",
            input={"messages": [{"role": "user",
                                 "content": "When Avengers Movie Got Released?"}]}
        )

        result1, result2 = await asyncio.gather(task1, task2)

        print(f"Result 1 [variant={result1.variant_name}]:")
        print(f"  {result1}")
        print()
        print(f"Result 2 [variant={result2.variant_name}]:")
        print(f"  {result2}")


if __name__ == "__main__":
    asyncio.run(run_parallel_inferences())