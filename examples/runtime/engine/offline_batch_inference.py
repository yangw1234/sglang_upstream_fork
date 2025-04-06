"""
Usage:
python3 offline_batch_inference.py  --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import dataclasses

import sglang as sgl
from sglang.srt.server_args import ServerArgs

conversations = [
    [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
    [
        {"role": "system", "content": "Always answer with Haiku"},
        {"role": "user", "content": "I am going to Paris, what should I see?"},
    ],
    [
        {"role": "system", "content": "Always answer with emojis"},
        {"role": "user", "content": "How to go from Beijing to NY?"},
    ],
    [
        {"role": "user", "content": "I am going to Paris, what should I see?"},
        {
            "role": "assistant",
            "content": "Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:\n\n1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.\n2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.\n3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.\n\nThese are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.",
        },
        {"role": "user", "content": "What is so great about #1?"},
    ],
]


def main(
    server_args: ServerArgs,
):
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # create a prompt using huggingface's chat template
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(server_args.model_path)

    prompt = tokenizer.apply_chat_template(conversations, tokenize=False)

    prompts.extend(prompt)
    # Create a sampling params object.
    sampling_params = {"temperature": 0.0, "top_p": 0.95, "max_new_tokens": 512}

    # Create an LLM.
    llm = sgl.Engine(**dataclasses.asdict(server_args))

    print("starting warmup")
    outputs = llm.generate(prompts, sampling_params)
    print("finished warmup")
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    main(server_args)
