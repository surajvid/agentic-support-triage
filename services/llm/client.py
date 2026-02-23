"""
LLM Client Wrapper

This isolates model configuration from agent logic.
If later you switch providers (OpenAI -> Azure OpenAI), you change it here.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def get_chat_model() -> ChatOpenAI:
    load_dotenv()

    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    # ChatOpenAI reads OPENAI_API_KEY from env automatically
    return ChatOpenAI(model=model_name, temperature=temperature)
