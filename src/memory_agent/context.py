"""Define the runtime context information for the agent."""

import os
from dataclasses import dataclass, field, fields

from typing_extensions import Annotated

from memory_agent import prompts


@dataclass(kw_only=True)
class Context:
    """Main context class for the memory graph system."""

    user_id: str = "default"
    """The ID of the user to remember in the conversation."""

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-sonnet-4-5-20250929",
        metadata={
            "description": "The name of the language model to use for the agent. "
            "Should be in the form: provider/model-name."
        },
    )

    system_prompt: str = prompts.SYSTEM_PROMPT

    def __post_init__(self) -> None:
        """
        dataclass 初始化后自动执行。

        遍历所有字段：
        - 如果字段仍使用默认值，则尝试读取同名的大写环境变量；
        - 如果环境变量存在，就用环境变量覆盖默认值；
        - 如果字段在创建 Context 时被显式传入，则不会被环境变量覆盖。

        优先级：显式传参 > 环境变量 > 默认值
        """
        for f in fields(self):
            if not f.init:
                continue

            if getattr(self, f.name) == f.default:
                setattr(self, f.name, os.environ.get(f.name.upper(), f.default))
