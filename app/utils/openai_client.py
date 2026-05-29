"""
Centralized OpenAI Client Initialization

Enterprise Benefits:
--------------------
- centralized provider configuration
- reusable embedding client
- easier provider replacement
- consistent initialization
"""

import os

from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings

from app.config.settings import EMBEDDING_MODEL


load_dotenv()


def get_embedding_model():

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=os.getenv("OPENAI_API_KEY")
    )