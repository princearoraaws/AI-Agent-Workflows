"""Adapt alternate OpenAI-compatible credentials for the book examples.

Python imports ``sitecustomize`` automatically when the repository is on
``PYTHONPATH``. This lets the unmodified examples use either NVIDIA NIM or the
custom OpenAI-compatible endpoint through the standard OpenAI environment
variables.
"""

import os

if os.environ.get("OPENAI_API_KEY"):
    # A native OpenAI key is already in the form the examples expect. Keeping
    # it unchanged also makes a deliberately configured OpenAI endpoint work.
    pass
elif os.environ.get("CUSTOM_LLM_API_KEY"):
    # The OpenAI Python client (used directly in a few examples) reads these
    # standard variables.
    os.environ["OPENAI_API_KEY"] = os.environ["CUSTOM_LLM_API_KEY"]
    os.environ.setdefault(
        "OPENAI_BASE_URL",
        os.environ.get("CUSTOM_LLM_BASE_URL", "https://litellm.ai.xxx/v1"),
    )
    if os.environ.get("CUSTOM_LLM_MODEL"):
        os.environ.setdefault("OPENAI_DEFAULT_MODEL", os.environ["CUSTOM_LLM_MODEL"])

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


elif os.environ.get("NVIDIA_API_KEY"):
    # The OpenAI Python client (used directly in a few examples) reads these
    # standard variables.
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
