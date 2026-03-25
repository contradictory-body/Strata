"""Compactor module for memory compaction operations."""

from agentscope.agent import ReActAgent
from agentscope.message import Msg
from agentscope.token import HuggingFaceTokenCounter

from ..utils import AsMsgHandler
from ....core.op import BaseOp
from ....core.utils import get_std_logger

logger = get_std_logger()


class Compactor(BaseOp):
    """Compactor class for compacting memory messages."""

    def __init__(
        self,
        memory_compact_threshold: int,
        token_counter: HuggingFaceTokenCounter,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.memory_compact_threshold: int = memory_compact_threshold

        self.msg_handler = AsMsgHandler(token_counter=token_counter)

    async def execute(self):
        messages: list[Msg] = self.context.get("messages", [])
        previous_summary: str = self.context.get("previous_summary", "")

        if not messages:
            return ""

        before_token_count = self.msg_handler.count_msgs_token(messages)
        history_formatted_str: str = self.msg_handler.format_msgs_to_str(
            messages=messages,
            memory_compact_threshold=self.memory_compact_threshold,
        )
        after_token_count = self.msg_handler.count_str_token(history_formatted_str)
        logger.info(f"Compactor before_token_count={before_token_count} after_token_count={after_token_count}")

        if not history_formatted_str:
            logger.warning(f"No history to compact. messages={messages}")
            return ""
        #这里用Agent而不是只用LLM原因如下：
        #整个项目建立在 AgentScope 的 Agent 体系上，所以这里也用统一的 Agent 方式来调用模型。
        #也就是说，它更像是：
        #为了统一接口
        #为了统一消息格式
        #为了统一 prompt/formatter 机制
        #为了和项目其他组件保持一致
        #而不是因为这里一定需要复杂的“先思考再工具调用”的长链路。
        agent = ReActAgent(#负责做 compact 摘要的智能体
            name="reme_compactor",
            model=self.as_llm,#负责做 compact 摘要的智能体使用的大模型
            sys_prompt=self.get_prompt("system_prompt"),#系统提示词，给这个智能体设定身份和工作规则
            formatter=self.as_llm_formatter,#格式器，负责把消息整理成模型更容易吃进去的格式
        )

        if previous_summary:
            prefix: str = self.get_prompt("update_user_message_prefix")
            suffix: str = self.get_prompt("update_user_message_suffix")
            user_message: str = (
                f"<conversation>\n{history_formatted_str}\n</conversation>\n\n"
                f"{prefix}\n\n"
                f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
                f"{suffix}"
            )
        else:
            user_message: str = f"<conversation>\n{history_formatted_str}\n</conversation>\n\n" + self.get_prompt(
                "initial_user_message",
            )
        logger.info(f"Compactor sys_prompt={agent.sys_prompt} user_message={user_message}")

        #构造一条消息 Msg(...)
        #这条消息的内容是 user_message
        #把它发给 agent.reply(...)
        #Agent 生成回复
        #返回的也是一个 Msg
        compact_msg: Msg = await agent.reply(
            Msg(
                name="reme",
                role="user",
                content=user_message,
            ),
        )

        history_compact: str = compact_msg.get_text_content()
        logger.info(f"Compactor Result:\n{history_compact}")
        return history_compact
