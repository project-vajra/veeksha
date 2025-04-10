from .base_llm_client import BaseLLMClient
from .openai_chat_completions_client import OpenAIChatCompletionsClient
from .openai_completions_client import OpenAICompletionsClient

SUPPORTED_APIS = ["openai_chat", "openai_completions"]


def construct_client(
    model_name: str,
    tokenizer_name: str,
    llm_api: str,
    api_url: str,
) -> BaseLLMClient:
    """Construct LLMClients that will be used to make requests to the LLM API.

    Args:
        llm_api: The name of the LLM API to use.

    Returns:
        The constructed LLMCLient

    """
    if llm_api == "openai_chat":
        impl = OpenAIChatCompletionsClient
    elif llm_api == "openai_completions":
        impl = OpenAICompletionsClient
    else:
        raise ValueError(
            f"llm_api must be one of the supported LLM APIs: {SUPPORTED_APIS}"
        )

    return impl(model_name, tokenizer_name, api_url)
