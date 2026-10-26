"""Configure the book examples for NVIDIA NIM when ``NVIDIA_API_KEY`` is set.

Python imports ``sitecustomize`` automatically when the repository is on
``PYTHONPATH``. This keeps the normal OpenAI configuration untouched and lets
text-based Agents SDK examples use NVIDIA's OpenAI-compatible Chat Completions
endpoint without changes to every chapter file.
"""

import os


if os.environ.get("NVIDIA_API_KEY"):
    # The OpenAI Python client (used directly in a few examples) reads these
    # standard variables. NVIDIA_API_KEY intentionally takes precedence: this
    # file is only active when the user has opted in to the NIM configuration.
    os.environ["OPENAI_API_KEY"] = os.environ["NVIDIA_API_KEY"]
    os.environ.setdefault("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1")

    # NIM's hosted LLM endpoint implements Chat Completions, while the Agents
    # SDK otherwise defaults to the OpenAI-only Responses API.
    try:
        from agents import set_default_openai_api, set_tracing_disabled
    except ModuleNotFoundError:
        # Let the importing program report its missing dependency normally.
        # This also keeps commands such as `python -m pip` usable before the
        # project requirements have been installed.
        pass
    else:
        set_default_openai_api("chat_completions")
        set_tracing_disabled(True)
