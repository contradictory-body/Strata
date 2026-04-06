"""Handler for AgentScope message processing, token counting, and context management."""

import json

from agentscope.message import Msg
from agentscope.token import HuggingFaceTokenCounter

from ....core.schema import AsMsgStat, AsBlockStat
from ....core.utils import get_std_logger

logger = get_std_logger()


class AsMsgHandler:
    """Handles token counting, formatting, and context compaction for AgentScope messages."""

    def __init__(self, token_counter: HuggingFaceTokenCounter):
        self._token_counter = token_counter

    #正常情况先把文本编码为token id序列，再数长度
    #失败则采用降级方案，用UTF-8字节数/4
    def count_str_token(self, text: str) -> int:
        """Count tokens in a string.

        Args:
            text: The text to count tokens for.

        Returns:
            The number of tokens in the text.
        """
        if not text:
            return 0

        try:
            token_ids = self._token_counter.tokenizer.encode(text)
            token_count = len(token_ids)
            return token_count

        except Exception as e:
            estimated_tokens = len(text.encode("utf-8")) // 4
            logger.warning(f"Failed to count string tokens: {text}, e={e}")
            return estimated_tokens

    #把工具结果统一转成“可读字符串 + 对应 token 数”
    #如果工具结果是纯文本，直接计数并保留
    #如果工具结果是block列表，则说明工具结果可能是多模态的，比如文本、图片、音频、视频、文件等，则采用下面的方案：
    #对于文本，直接计数并保留
    #对于其它，一般有两种保存类型，一种是URL，就是链接，一种是base64，就是把二进制文件内容，编码成一大串普通字符
    #假设保存类型是URL，它一般比较短，因此直接做token统计，先保留一个说明，比如[image] https://example.com/a.png，然后统计这个说明的token。
    #假设保存类型是base64，因为它很长，所以不tokenizer，而是按长度粗略估计它的token量，也就是UTF-8字节数/4
    #假设保存类型是文字，它会记录一个文件说明：[file] {file_name}: {file_path}，然后将file_path tokenizer，然后计算token
    def _format_tool_result_output(self, output: str | list[dict]) -> tuple[str, int]:
        """Convert tool result output to string."""
        if isinstance(output, str):
            return output, self.count_str_token(output)

        textual_parts = []
        total_token_count = 0
        for block in output:
            try:
                if not isinstance(block, dict) or "type" not in block:
                    logger.warning(
                        "Invalid block: %s, expected a dict with 'type' key, skipped.",
                        block,
                    )
                    continue

                block_type = block["type"]

                if block_type == "text":
                    textual_parts.append(block.get("text", ""))
                    total_token_count += self.count_str_token(textual_parts[-1])

                elif block_type in ["image", "audio", "video"]:
                    source = block.get("source", {})
                    if source.get("type") == "base64":
                        #当type为base64时，不会添加可读字符串
                        data = source.get("data", "")
                        total_token_count += len(data) // 4 if data else 10
                    else:
                        url = source.get("url", "")
                        total_token_count += self.count_str_token(url) if url else 10
                        textual_parts.append(f"[{block_type}] {url}")

                elif block_type == "file":
                    file_path = block.get("path", "") or block.get("url", "")
                    file_name = block.get("name", file_path)
                    textual_parts.append(f"[file] {file_name}: {file_path}")
                    total_token_count += self.count_str_token(file_path)

                else:
                    logger.warning(
                        "Unsupported block type '%s' in tool result, skipped.",
                        block_type,
                    )

            except Exception as e:
                logger.warning(
                    "Failed to process block %s: %s, skipped.",
                    block,
                    e,
                )

        return "\n".join(textual_parts), total_token_count

    #功能是：把一条消息变成结构化统计对象
    #具体而言，就是把原始 Msg 解析成一个统一的统计对象 AsMsgStat，其中每个内容块会变成AsBlockStat，相当于给消息做一个标准化
    #说人话就是：把各种乱七八糟的消息内容，统一翻译成 ReMe 自己看得懂的统计格式。
    #先区分是字符串类型还是list类型，字符串类型的话就直接创建text block，list类型（block列表）的话就按照text、thinking、image/audio/video、tool_use、tool_result分别处理
    def stat_message(self, message: Msg) -> AsMsgStat:
        """Analyze a message and generate block statistics."""
        blocks = []
        if isinstance(message.content, str):
            blocks.append(
                AsBlockStat(
                    block_type="text",
                    text=message.content,
                    token_count=self.count_str_token(message.content),
                ),
            )
            return AsMsgStat(
                name=message.name or message.role,
                role=message.role,
                content=blocks,
                timestamp=message.timestamp or "",
                metadata=message.metadata or {},
            )

        if not isinstance(message.content, list):
            logger.warning(
                "Unexpected message.content type %s, expected str or list, returning empty stat.",
                type(message.content),
            )
            return AsMsgStat(
                name=message.name or message.role,
                role=message.role,
                content=blocks,
                timestamp=message.timestamp or "",
                metadata=message.metadata or {},
            )

        for block in message.content:
            block_type = block.get("type", "unknown")

            if block_type == "text":
                text = block.get("text", "")
                token_count = self.count_str_token(text)
                blocks.append(
                    AsBlockStat(
                        block_type=block_type,
                        text=text,
                        token_count=token_count,
                    ),
                )

            elif block_type == "thinking":
                thinking = block.get("thinking", "")
                token_count = self.count_str_token(thinking)
                blocks.append(
                    AsBlockStat(
                        block_type=block_type,
                        text=thinking,
                        token_count=token_count,
                    ),
                )

            elif block_type in ("image", "audio", "video"):
                source = block.get("source", {})
                url = source.get("url", "")
                if source.get("type") == "base64":
                    data = source.get("data", "")
                    token_count = len(data) // 4 if data else 10
                else:
                    token_count = self.count_str_token(url) if url else 10
                blocks.append(
                    AsBlockStat(
                        block_type=block_type,
                        text="",
                        token_count=token_count,
                        media_url=url,
                    ),
                )

            elif block_type == "tool_use":
                tool_name = block.get("name", "")
                tool_input = block.get("input", "")
                try:
                    input_str = json.dumps(tool_input, ensure_ascii=False)
                except (TypeError, ValueError):
                    input_str = str(tool_input)
                token_count = self.count_str_token(tool_name + input_str)
                blocks.append(
                    AsBlockStat(
                        block_type=block_type,
                        text="",
                        token_count=token_count,
                        tool_name=tool_name,
                        tool_input=input_str,
                    ),
                )

            elif block_type == "tool_result":
                tool_name = block.get("name", "")
                output = block.get("output", "")
                formatted_output, token_count = self._format_tool_result_output(output)
                blocks.append(
                    AsBlockStat(
                        block_type=block_type,
                        text="",
                        token_count=token_count,
                        tool_name=tool_name,
                        tool_output=formatted_output,
                    ),
                )

            else:
                logger.warning("Unsupported block type %s, skipped.", block_type)

        return AsMsgStat(
            name=message.name or message.role,
            role=message.role,
            content=blocks,
            timestamp=message.timestamp or "",
            metadata=message.metadata or {},
        )

    #统计一组消息的总token
    #每条消息先用上面的stat_message函数计算total_tokens，然后求和
    def count_msgs_token(self, messages: list[Msg]) -> int:
        """Count total token count of a list of messages."""
        return sum(self.stat_message(msg).total_tokens for msg in messages)

    #把消息列表转成一段字符串，但只保留“从最新往前、在阈值内能放得下的部分”。
    #这个函数在压缩消息的时候，也就是compactor.py中才被使用，它与分隔哪些消息被压缩，哪些被保留这个步骤无关。
    #它的功能是负责“挑出一段适合被压缩的最近历史，并把它格式化成字符串”。超过阈值的部分则被舍弃，不进行压缩。
    def format_msgs_to_str(
        self,
        messages: list[Msg],
        memory_compact_threshold: int,
        include_thinking: bool = False,
    ) -> str:
        """Format list of messages to a single formatted string.

        Messages are processed in reverse order (newest first) and older
        messages are skipped when token count exceeds memory_compact_threshold.

        Args:
            messages: List of Msg objects to format.
            memory_compact_threshold: Maximum token count before skipping older messages.
            include_thinking: Whether to include thinking blocks in output.
        """
        if not messages:
            return ""

        formatted_parts: list[str] = []
        total_token_count = 0

        #关键点：从后往前处理，也就是优秀保留最近信息
        for i in range(len(messages) - 1, -1, -1):
            stat = self.stat_message(messages[i])#先把消息标准化
            formatted_content = stat.format(include_thinking=include_thinking)#再格式化成字符串
            content_token_count = self.count_str_token(formatted_content)#再数这个字符串的token
            #stat_message的返回结果中已经包含了token量，为什么还需要重新计算呢？
            #因为它计算的是“内容级token”，回答的是“这条消息本身大概有多大”
            #而格式化后重新计算的token数回答的是“把这条消息整理成最终字符串，真正塞进后续文本上下文里，会占多少 token”
            #stat_message() 统计的是结构化内容字段，format() 之后得到的是可读字符串表示

            is_latest = i == len(messages) - 1
            #超阈值时跳过更老消息
            if not is_latest and total_token_count + content_token_count > memory_compact_threshold:
                logger.info(
                    "Skipping older messages: adding %d tokens would exceed threshold %d (current: %d)",
                    content_token_count,
                    memory_compact_threshold,
                    total_token_count,
                )
                break
            #即使最新消息单条就超，也照样保留
            if is_latest and content_token_count > memory_compact_threshold:
                logger.warning(
                    "Latest message alone (%d tokens) exceeds threshold %d, including it anyway.",
                    content_token_count,
                    memory_compact_threshold,
                )

            formatted_parts.append(formatted_content)
            total_token_count += content_token_count

        formatted_parts.reverse()
        return "\n\n".join(formatted_parts)

    #检查工具调用链是否完整
    #具体来说，它收集所有 tool_use 的 id，收集所有 tool_result 的 id，比较两个集合是否相等
    @staticmethod
    def validate_tool_ids_alignment(messages: list[Msg]) -> bool:
        """Check if tool_use_ids and tool_result_ids are properly aligned.

        Args:
            messages: List of Msg objects to validate.

        Returns:
            True if all tool_use ids have corresponding tool_result ids and vice versa.
        """
        tool_use_ids: set[str] = set()
        tool_result_ids: set[str] = set()

        for msg in messages:
            for block in msg.get_content_blocks("tool_use"):
                if tool_id := block.get("id"):
                    tool_use_ids.add(tool_id)
            for block in msg.get_content_blocks("tool_result"):
                if tool_id := block.get("id"):
                    tool_result_ids.add(tool_id)

        return tool_use_ids == tool_result_ids

    def context_check(
        self,
        messages: list[Msg],
        memory_compact_threshold: int,
        memory_compact_reserve: int,
    ) -> tuple[list[Msg], list[Msg], bool]:
        """Check if context exceeds threshold and split messages accordingly.

        Only when total tokens exceed memory_compact_threshold, messages are split into
        messages_to_keep (within reserve limit) and messages_to_compact (older messages).

        Args:
            messages: List of Msg objects to check.
            memory_compact_threshold: Maximum token count threshold to trigger compaction.
            memory_compact_reserve: Token limit for messages to keep.

        Returns:
            A tuple of (messages_to_compact, messages_to_keep, tools_aligned):
            - messages_to_compact: Older messages that exceed reserve limit
            - messages_to_keep: Recent messages within the reserve limit
            - tools_aligned: Whether tool_use and tool_result ids are aligned in messages_to_keep
        """
        if not messages:
            return [], [], True#空消息直接返回

        # Calculate total tokens and stats for all messages
        msg_stats: list[tuple[Msg, AsMsgStat]] = []
        total_tokens = 0
        for msg in messages:
            stat = self.stat_message(msg)#为每条消息生成标准对象
            msg_stats.append((msg, stat))
            total_tokens += stat.total_tokens#统计总token数

        # If total tokens don't exceed threshold, no split needed
        if total_tokens < memory_compact_threshold:#小于阈值则无需压缩
            return [], messages, True

        # Collect all tool_use ids and their message indices
        # tool_use_id -> message index
        tool_use_locations: dict[str, int] = {}#记录tool_use的位置
        # tool_result_id -> message index
        tool_result_locations: dict[str, int] = {}#记录tool_result的位置

        for idx, (msg, _) in enumerate(msg_stats):
            for block in msg.get_content_blocks("tool_use"):
                tool_id = block.get("id", "")
                if tool_id:
                    tool_use_locations[tool_id] = idx

            for block in msg.get_content_blocks("tool_result"):
                tool_id = block.get("id", "")
                if tool_id:
                    tool_result_locations[tool_id] = idx

        # Iterate from the end, accumulating messages to keep within reserve limit
        keep_indices: set[int] = set()
        accumulated_tokens = 0

        for i in range(len(msg_stats) - 1, -1, -1):
            # Skip messages already added as tool_use dependencies to avoid double-counting tokens
            if i in keep_indices:
                continue

            msg, stat = msg_stats[i]

            # Check if adding this message would exceed reserve limit
            #threshold和reverse的区别：
            #threshold决定是否需要压缩。消息token数超过threshold就要启动压缩，否则不需要
            #revers决定压缩时，最多能够保留多少token不压缩，由于保留顺序是从新往旧，也就是说reverse决定“压缩时，最近消息最多保留多少 token”
            if accumulated_tokens + stat.total_tokens > memory_compact_reserve:
                logger.info(
                    "Context check: adding message %d with %d tokens would exceed reserve %d (current: %d)",
                    i,
                    stat.total_tokens,
                    memory_compact_reserve,
                    accumulated_tokens,
                )
                break

            # Check tool_result dependencies - if this message has tool_result,
            # we need to ensure the corresponding tool_use is also included
            #如果消息里有 tool_result，检查它依赖的 tool_use
            tool_result_ids = [
                block.get("id", "") for block in msg.get_content_blocks("tool_result") if block.get("id", "")
            ]

            # Calculate extra tokens needed for dependent tool_use messages
            #意思是：
            #如果我要保留这个 tool_result
            #那我还得额外把对应的 tool_use 也保留下来
            #这些依赖消息也占 token，需要一起算进去
            extra_tokens = 0
            dependent_indices: set[int] = set()

            for tool_id in tool_result_ids:
                if tool_id in tool_use_locations:
                    tool_use_idx = tool_use_locations[tool_id]
                    if tool_use_idx not in keep_indices and tool_use_idx != i:
                        dependent_indices.add(tool_use_idx)
                        _, dep_stat = msg_stats[tool_use_idx]
                        extra_tokens += dep_stat.total_tokens

            # Check if we can fit this message plus its dependencies within reserve
            #如果当前消息 + 依赖消息一并加入会超 reserve，就停止
            if accumulated_tokens + stat.total_tokens + extra_tokens > memory_compact_reserve:
                logger.info(
                    "Context check: message %d requires %d extra tokens for tool_use dependencies, "
                    "total would exceed reserve %d",
                    i,
                    extra_tokens,
                    memory_compact_reserve,
                )
                break

            #把当前消息和依赖一起加入保留集合
            # Add this message and its dependencies
            keep_indices.add(i)
            keep_indices.update(dependent_indices)
            accumulated_tokens += stat.total_tokens + extra_tokens
        
        #根据 keep_indices 重建两组消息，按原顺序重建的，所以不会把消息顺序打乱
        # Build final lists based on keep_indices (preserve original order)
        messages_to_compact = []
        messages_to_keep = []

        for idx, (msg, _) in enumerate(msg_stats):
            if idx in keep_indices:
                messages_to_keep.append(msg)
            else:
                messages_to_compact.append(msg)

        # Validate tool ids alignment for messages_to_keep
        tools_aligned = self.validate_tool_ids_alignment(messages_to_keep)#最终检查工具调用链是否对齐

        logger.info(
            "Context check result: %d messages to compact, %d messages to keep, "
            "total tokens: %d, threshold: %d, reserve: %d, kept tokens: %d, "
            "tools_aligned: %s",
            len(messages_to_compact),
            len(messages_to_keep),
            total_tokens,
            memory_compact_threshold,
            memory_compact_reserve,
            accumulated_tokens,
            tools_aligned,
        )

        return messages_to_compact, messages_to_keep, tools_aligned#记录日志并返回
