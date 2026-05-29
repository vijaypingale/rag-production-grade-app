"""DISABLED: Centralized AWS Bedrock Embedding Client

Note: Bedrock support is temporarily disabled in favor of OpenAI.
This module will be re-enabled later. All Bedrock calls should go through
the provider abstraction in app/embeddings/embedding_generator.py.

When re-enabling Bedrock:
1. Update .env with valid AWS credentials
2. Set EMBEDDING_PROVIDER=bedrock in .env
3. Ensure the Bedrock model is enabled in your AWS account

Original documentation:
-----------------------

Enterprise Benefits:
--------------------
- centralized provider configuration
- reusable embedding client
- easier provider replacement (Bedrock <-> OpenAI <-> Cohere)
- consistent initialization across services
- single place to inject AWS credentials / region

Why Bedrock for embeddings:
---------------------------
- managed AWS service (no key management for OpenAI)
- private VPC connectivity available (no data leaves AWS)
- IAM-based access control (fits enterprise security model)
- multiple model families behind one API (Titan, Cohere, etc.)
- pay-per-token, no infrastructure to manage

Required environment variables:
-------------------------------
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_REGION                     (defaults to us-east-1)
- BEDROCK_EMBEDDING_MODEL        (defaults to amazon.titan-embed-text-v2:0)

Auth alternatives:
------------------
On EC2/ECS/EKS you should prefer an IAM role attached to the
compute, not static keys. The boto3 client used internally by
langchain-aws will pick up role credentials automatically.
"""

import os

from dotenv import load_dotenv

from langchain_aws import BedrockEmbeddings

from app.config.settings import (
    BEDROCK_EMBEDDING_MODEL,
    BEDROCK_REGION,
)


# Load .env so local dev picks up AWS credentials.
# In production, credentials come from the instance role
# and load_dotenv() is effectively a no-op.
load_dotenv()


def get_embedding_model():
    """
    Return a configured Bedrock embedding model client.

    Returns:
    --------
    langchain_aws.BedrockEmbeddings
        Ready-to-use embedding model. Exposes:
        - embed_documents(list[str]) -> list[list[float]]
        - embed_query(str)           -> list[float]

    Why a factory function (not a module-level singleton):
    ------------------------------------------------------
    - tests can mock it cleanly
    - allows lazy initialization (no AWS call at import time)
    - lets us swap providers without rewriting callers
    """

    # BedrockEmbeddings auto-resolves credentials in this order:
    #   1. explicit kwargs (not used here)
    #   2. AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars
    #   3. ~/.aws/credentials  (local dev)
    #   4. instance role       (production EC2/ECS/EKS)
    return BedrockEmbeddings(
        model_id=BEDROCK_EMBEDDING_MODEL,
        region_name=BEDROCK_REGION,
    )
