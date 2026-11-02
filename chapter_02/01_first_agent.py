# Step 1: Import the Agents SDK building blocks. No API client is created and
# no network request is sent by this import.
from agents import Agent, Runner
from dotenv import load_dotenv

# Step 2: Load settings from .env into environment variables, including the
# API key and default model. The SDK reads these later, when it needs a model.
load_dotenv()

# Step 3: Define the system instructions. They tell the LLM how to behave and
# will become the system message in the eventual API request.
instructions = """
You are a research planning assistant.

**TASK INSTRUCTIONS**
- You will be given a research topic.
- Your task is to provide a plan on how to research this topic.
- Output 5 concise tasks (5 words or less) to your plan.
"""

# Step 4: Define an agent configuration. This only stores its name and
# instructions; it does not create a client or call an LLM yet.
agent = Agent(
    name="Research Planner",
    instructions=instructions,
)

# Step 5: Define the user's message that will be sent to the agent.
input = "learn about AI agents"

# Step 6: Start the agent run and wait for it to finish. Inside the SDK,
# Runner resolves the model/provider, lazily creates an AsyncOpenAI client
# using the environment variables, builds system + user messages, sends the
# HTTP request, and returns a result object. This example has no tools or
# handoffs, so it normally makes one LLM request.
result = Runner.run_sync(
    agent,
    input=input,
)

# Step 7: Extract the final assistant text from the result and display it.
print(result.final_output)
