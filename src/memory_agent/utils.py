"""Utility functions used in our graph."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """

    # # 将 "provider/model" 格式的模型名称拆成 provider 和 model，
    # # 再通过 LangChain 的 init_chat_model 创建统一的聊天模型对象。
    provider, model = fully_specified_name.split("/", maxsplit=1)
    return init_chat_model(model, model_provider=provider)
