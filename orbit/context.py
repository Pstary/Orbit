"""Multi-layer context compression.

Claude Code uses a 4-layer strategy:
  1. HISTORY_SNIP   - trim old tool outputs to a one-line summary
  2. Microcompact   - LLM-powered summary of old turns (cached)
  3. CONTEXT_COLLAPSE - aggressive compression when nearing hard limit
  4. Autocompact    - periodic background compaction

Orbit implements the same idea in 3 layers:
  Layer 1 (tool_snip)   - replace verbose tool results with truncated versions
  Layer 2 (summarize)   - LLM-powered summary of old conversation
  Layer 3 (hard_collapse) - last resort: drop everything except summary + recent
- tool_snip ：上下文超过50%时，优先裁剪过长的工具输出，只保留关键开头和结尾，低成本释放上下文空间。
- summarize ：上下文超过70%时，用LLM把旧对话压缩成摘要，同时保留最近几轮原始消息，避免丢失当前任务状态。
- hard_collapse ：上下文超过90%时，执行兜底压缩，只保留旧历史摘要和最近消息，防止超过模型上下文上限。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llm import LLM

# 用字符数粗略估算token数。这里假设中英文混合内容大约 3 个字符约等于 1 个token。它不是精确tokenizer，只是便宜快速的估算。
def _approx_tokens(text: str) -> int:
    """Rough token count, roughly 3 chars per token for mixed en/zh content."""
    return len(text) // 3

# 这个函数统计整段对话历史大概用了多少token。
def estimate_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        if m.get("content"):
            total += _approx_tokens(m["content"])
        if m.get("tool_calls"):
            total += _approx_tokens(str(m["tool_calls"]))
    return total


class ContextManager:
    def __init__(self, max_tokens: int = 128_000):
        self.max_tokens = max_tokens
        # layer thresholds (fraction of max_tokens)
        self._snip_at = int(max_tokens * 0.50)    # 50% -> snip tool outputs
        self._summarize_at = int(max_tokens * 0.70)  # 70% -> LLM summarize
        self._collapse_at = int(max_tokens * 0.90)   # 90% -> hard collapse

    def maybe_compress(self, messages: list[dict], llm: LLM | None = None) -> bool:
        """Apply compression layers as needed. Returns True if any compression happened."""
        current = estimate_tokens(messages)
        compressed = False

        # Layer 1: snip verbose tool outputs
        if current > self._snip_at and self._snip_tool_outputs(messages):
            compressed = True
            # 重新计算当前token数
            current = estimate_tokens(messages)

        # Layer 2: LLM-powered summarization of old turns
        if current > self._summarize_at and len(messages) > 10 and self._summarize_old(messages, llm, keep_recent=8):
            compressed = True
            current = estimate_tokens(messages)

        # Layer 3: hard collapse - last resort
        if current > self._collapse_at and len(messages) > 4:
            self._hard_collapse(messages, llm)
            compressed = True

        return compressed

    # 这段代码遍历所有tool消息，把超过1500字符且超过6行的工具输出替换成“前3行+省略提示+后3行”，用最低成本释放上下文空间。
    @staticmethod
    def _snip_tool_outputs(messages: list[dict]) -> bool:
        """Layer 1: Truncate tool results over 1500 chars to their first/last lines.

        This mirrors Claude Code's HISTORY_SNIP which replaces old tool outputs
        with a one-line summary to reclaim context space.
        """
        changed = False
        for m in messages:
            if m.get("role") != "tool":
                continue
            content = m.get("content", "")
            if len(content) <= 1500:
                continue
            lines = content.splitlines()
            if len(lines) <= 6:
                continue
            # keep first 3 + last 3 lines
            snipped = (
                "\n".join(lines[:3])
                + f"\n... ({len(lines)} lines, snipped to save context) ...\n"
                + "\n".join(lines[-3:])
            )
            m["content"] = snipped
            changed = True
        return changed

    @staticmethod
    def _safe_split(messages: list[dict], keep_recent: int) -> int:
        """Index where the kept tail should start.

        Walk the boundary back so a 'tool' result is never separated from the
        assistant message whose tool_calls produced it - an orphaned tool
        message has no preceding tool_calls and OpenAI-compatible APIs reject it.
        """
        #在压缩历史前找一个不破坏tool_call配对关系的切分点，保证保留下来的最近消息不会从孤立的tool消息开始。
        split = max(0, len(messages) - keep_recent)
        while split > 0 and messages[split].get("role") == "tool":
            split -= 1
        return split

    # 在上下文超过70%且消息数超过10条时，把旧消息总结掉，通常保留最近8条原始消息，但会为了tool_call配对安全而可能多保留几条。
    def _summarize_old(self, messages: list[dict], llm: LLM | None,
                       keep_recent: int = 8) -> bool:
        """Layer 2: Summarize old conversation, keep recent messages intact."""
        if len(messages) <= keep_recent:
            return False

        split = self._safe_split(messages, keep_recent)
        old = messages[:split]
        tail = messages[split:]

        summary = self._get_summary(old, llm)

        messages.clear()
        messages.append({
            "role": "user",
            "content": f"[Context compressed - conversation summary]\n{summary}",
        })
        messages.append({
            "role": "assistant",
            "content": "Got it, I have the context from our earlier conversation.",
        })
        messages.extend(tail)
        return True

    def _hard_collapse(self, messages: list[dict], llm: LLM | None):
        """Layer 3: Emergency compression. Keep only last 4 messages + summary."""
        split = self._safe_split(messages, 4 if len(messages) > 4 else 2)
        tail = messages[split:]
        summary = self._get_summary(messages[:split], llm)

        messages.clear()
        messages.append({
            "role": "user",
            "content": f"[Hard context reset]\n{summary}",
        })
        messages.append({
            "role": "assistant",
            "content": "Context restored. Continuing from where we left off.",
        })
        messages.extend(tail)

    def _get_summary(self, messages: list[dict], llm: LLM | None) -> str:
        """Generate summary via LLM or fallback to extraction."""
        # 把多条messages压平成一段普通文本，方便拿去总结。
        flat = self._flatten(messages)

        if llm:
            try:
                # 这里专门给模型一个system prompt，要求它压缩对话，并保留这些关键信息
                # 改过哪些文件
                # 做过哪些关键决策
                # 遇到过哪些错误
                # 当前任务状态
                resp = llm.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Compress this conversation into a brief summary. "
                                "Preserve: file paths edited, key decisions made, "
                                "errors encountered, current task state. "
                                "Drop: verbose command output, code listings, "
                                "redundant back-and-forth."
                            ),
                        },
                        # 最多只拿前15000个字符去总结，避免摘要请求本身也太长
                        {"role": "user", "content": flat[:15000]},
                    ],
                )
                return resp.content
            except Exception:  # noqa: BLE001, S110
                # summarization is best-effort; fall back to extraction below
                pass

        # fallback: extract key lines
        # 直接从历史消息里提取文件路径、error行等关键信息。
        return self._extract_key_info(messages)

    @staticmethod
    def _flatten(messages: list[dict]) -> str:
        parts = []
        for m in messages:
            role = m.get("role", "?")
            text = m.get("content", "") or ""
            if text:
                parts.append(f"[{role}] {text[:400]}")
        return "\n".join(parts)

    @staticmethod
    def _extract_key_info(messages: list[dict]) -> str:
        """Fallback: extract file paths, errors, and decisions without LLM."""
        import re
        files_seen = set()
        errors = []

        for m in messages:
            text = m.get("content", "") or ""
            # extract file paths
            for match in re.finditer(r'[\w./\-]+\.\w{1,5}', text):
                files_seen.add(match.group())
            # extract error lines
            for line in text.splitlines():
                if "error" in line.lower():
                    errors.append(line.strip()[:150])

        parts = []
        if files_seen:
            parts.append(f"Files touched: {', '.join(sorted(files_seen)[:20])}")
        if errors:
            parts.append(f"Errors seen: {'; '.join(errors[:5])}")
        return "\n".join(parts) or "(no extractable context)"
