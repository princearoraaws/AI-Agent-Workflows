"""The first-agent example, expanded into one explicit asynchronous LLM request.

This version shows the async work that Runner.run_sync() hides in the original
example. It does not use Agent or Runner.

Original-code mapping (01_first_agent.py):
* Original line 5, ``load_dotenv()``, is this file's Step 1.
* Original lines 8-22, instructions + ``Agent(...)``, are this file's Step 3.
* Original lines 24-27, ``Runner.run_sync(...)``, hide this file's Steps 2,
  4, 5, and 7: setup, the async client/request, and event-loop management.
* Original line 29, ``print(result.final_output)``, is this file's Step 6.
"""

import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam


# Step 1 — original line 5: Load .env variables into os.environ.
load_dotenv()


# Step 2 — hidden inside original Runner.run_sync(...): Choose one provider
# and model, then translate its configuration to standard OpenAI-client inputs.
# The order matches sitecustomize.py.
provider_name: str
api_key: str
base_url: str | None
model: str | None

if os.getenv("OPENAI_API_KEY"):
    provider_name = "OpenAI"
    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5.1")
elif os.getenv("CUSTOM_LLM_API_KEY"):
    provider_name = "Custom OpenAI-compatible LLM"
    api_key = os.environ["CUSTOM_LLM_API_KEY"]
    base_url = os.getenv(
        "CUSTOM_LLM_BASE_URL", "https://litellm.ai.xxx/v1"
    )
    model = os.getenv("CUSTOM_LLM_MODEL") or os.getenv("OPENAI_DEFAULT_MODEL")
elif os.getenv("NVIDIA_API_KEY"):
    provider_name = "NVIDIA NIM"
    api_key = os.environ["NVIDIA_API_KEY"]
    base_url = "https://integrate.api.nvidia.com/v1"
    model = os.getenv("OPENAI_DEFAULT_MODEL")
else:
    raise RuntimeError(
        "Set exactly one of OPENAI_API_KEY, CUSTOM_LLM_API_KEY, or NVIDIA_API_KEY in .env."
    )

if not model:
    raise RuntimeError("Set a model name for the selected provider in .env.")
# Keep a non-optional value for use inside the nested async function. Static
# checkers cannot assume that the outer `model` variable remains unchanged.
model_name: str = model


# Step 3 — original lines 8-22: Define the system instructions and user input.
# Agent(...) stores these values; here we convert them directly to API messages.
instructions = """
You are a research planning assistant.

TASK INSTRUCTIONS
- You will be given a research topic.
- Your task is to provide a plan on how to research this topic.
- Output 5 concise tasks (5 words or less) to your plan.
"""
user_input = "learn about AI agents"

messages: list[ChatCompletionMessageParam] = [
    {"role": "system", "content": instructions},
    {"role": "user", "content": user_input},
]


async def main() -> None:
    """Create an async client and wait for one LLM response."""
    # Step 4 — hidden inside original Runner.run_sync(...): Create an async
    # client. `async with` closes its HTTP resources after the request.
    async with AsyncOpenAI(api_key=api_key, base_url=base_url) as client:
        # Step 5 — hidden inside original Runner.run_sync(...): Send the HTTP
        # request without blocking the event loop, then wait at `await`.
        print(f"Sending a request to {provider_name} using model: {model_name}")
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
        )

    # Step 6 — original line 29: Extract and print the final assistant text.
    final_output = response.choices[0].message.content
    print("\nLLM response:\n")
    print(final_output)


# Step 7 — hidden inside original Runner.run_sync(...): Create an event loop,
# run main() to completion, then close it.
asyncio.run(main())
