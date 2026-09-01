"""Tests for the persistent memory module.

覆盖范围：
- 存储层：写入/读取/索引、路径逃逸防护、frontmatter(updated 字段)、去重/校验；
- 加固层：exclusive_file_lock 跨平台互斥、atomic_write_text 原子写（无临时文件残留）；
- 注入保护：MEMORY.md 规则截断（200行/25KB + WARNING）、30 天陈旧记忆老化警告、
  记忆策略行注入；
- write_memory_file 受限工具：参数校验 + 乐观锁去重 + 原子写 + 重建索引；
- forked agent 提取：同步入口（测试用）/ fire-and-forget 异步入口 / ScriptedLLM 跳过。
"""

import os
import time

import pytest

from orbit.llm import LLMResponse, ScriptedLLM, ToolCall
from orbit.memory import (
    MEMORY_INDEX_NAME,
    MemoryManager,
    MemoryStore,
    atomic_write_text,
    exclusive_file_lock,
    memory_slug,
    truncate_index,
)
from orbit.tools.memory_tool import WriteMemoryFileTool


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path)


def _write(store, name="python-style", mem_type="feedback", description="Use ruff format",
           body="Always run ruff format before committing."):
    store.write_record(name, mem_type, description, body)


# ---------------------------------------------------------------------------
# 基础存储层
# ---------------------------------------------------------------------------

def test_memory_slug_keeps_cjk():
    assert memory_slug("部署流程") == "部署流程"
    assert memory_slug("Python Style!") == "python-style"


def test_write_and_list_roundtrip(store):
    _write(store)
    records = store.list_records()
    assert len(records) == 1
    record = records[0]
    assert record["name"] == "python-style"
    assert record["type"] == "feedback"
    assert record["description"] == "Use ruff format"
    assert "ruff format" in record["body"]
    assert record["filename"] == "python-style.md"
    # 新格式 frontmatter 自带 updated 日期（ISO 格式）。
    assert record["updated"]


def test_rebuild_index_lists_records(store):
    _write(store)
    index = store.read_index()
    assert "python-style.md" in index
    assert "Use ruff format" in index
    assert (store.memory_dir / MEMORY_INDEX_NAME).exists()


def test_read_record(store):
    _write(store)
    content = store.read_record("python-style.md")
    assert content is not None
    assert "Always run ruff format" in content
    assert store.read_record("missing.md") is None


def test_path_escape_rejected(store):
    with pytest.raises(ValueError):
        store._record_path("../evil.md")
    with pytest.raises(ValueError):
        store._record_path("nested/evil.md")


def test_should_store_rejects_duplicates(store):
    _write(store)
    existing = store.list_records()
    # same slug (different casing/whitespace) -> duplicate
    assert not store.should_store(
        {"scope": "persistent", "type": "project", "name": "Python Style",
         "description": "new desc", "body": "new body"}, existing)
    # same normalized body -> duplicate
    assert not store.should_store(
        {"scope": "persistent", "type": "project", "name": "other-name",
         "description": "other desc", "body": "  always run ruff format before committing.  "}, existing)
    # fresh durable record accepted
    assert store.should_store(
        {"scope": "persistent", "type": "user", "name": "language",
         "description": "Prefers Chinese replies", "body": "Reply in Chinese by default."}, existing)


def test_should_store_rejects_temporary_and_non_persistent(store):
    base = {"type": "project", "name": "tmp", "description": "d", "body": "b"}
    # current_task scope is not persisted
    assert not store.should_store({**base, "scope": "current_task"}, [])
    # temporary markers (English + Chinese)
    assert not store.should_store(
        {**base, "scope": "persistent", "body": "Use port 8000 for this session only"}, [])
    assert not store.should_store(
        {**base, "scope": "persistent", "name": "临时路径", "description": "暂时用这个目录",
         "body": "暂时放在 tmp 目录"}, [])
    # invalid type
    assert not store.should_store(
        {**base, "scope": "persistent", "type": "weird"}, [])


def test_validate_record():
    good = {"name": "n", "type": "user", "description": "d", "body": "b", "scope": "persistent"}
    validated = MemoryStore.validate_record(good, require_scope=True)
    assert validated is not None and validated["scope"] == "persistent"
    assert MemoryStore.validate_record({"name": "n", "type": "bad", "description": "d", "body": "b"}) is None
    assert MemoryStore.validate_record("not a dict") is None


def test_keyword_selection_matches_chinese(store):
    store.write_record("部署流程", "project", "生产环境发布步骤", "先跑测试再发版到生产环境。")
    store.write_record("编码风格", "feedback", "Python 代码风格要求", "使用 ruff 格式化。")
    records = store.list_records()
    selected = MemoryManager._keyword_selection(records, "生产环境怎么部署？")
    assert selected == ["部署流程.md"]


# ---------------------------------------------------------------------------
# 文件锁 + 原子写入
# ---------------------------------------------------------------------------

def test_atomic_write_text_replaces_and_leaves_no_temp_files(tmp_path):
    target = tmp_path / "mem" / "a.md"
    atomic_write_text(target, "hello\n")
    assert target.read_text(encoding="utf-8") == "hello\n"
    # 成功写入后同目录不应残留 .tmp 临时文件。
    assert list((tmp_path / "mem").glob("*.tmp")) == []
    # 覆盖写：内容完整替换，仍然无临时文件残留。
    atomic_write_text(target, "world\n")
    assert target.read_text(encoding="utf-8") == "world\n"
    assert list((tmp_path / "mem").glob("*.tmp")) == []


def test_exclusive_file_lock_blocks_concurrent_holder(tmp_path):
    lock_path = tmp_path / ".memory.lock"
    with exclusive_file_lock(lock_path):
        # 持锁期间，另一处用极短超时抢锁必须超时失败（锁是互斥的）。
        with pytest.raises(TimeoutError):
            with exclusive_file_lock(lock_path, timeout=0.3):
                pytest.fail("should not acquire while lock is held")
    # 锁释放后可以重新获取。
    with exclusive_file_lock(lock_path, timeout=1.0):
        pass


def test_store_record_optimistic_lock_skips_duplicate(store):
    candidate = {
        "scope": "persistent", "name": "reply-language", "type": "user",
        "description": "User prefers Chinese replies",
        "body": "Always reply in Chinese by default.",
    }
    # 第一次写入成功（返回路径）。
    path = store.store_record(dict(candidate))
    assert path is not None and path.exists()
    # 第二次写入等价内容：锁内重读发现重复，乐观锁跳过，返回 None，不产生新文件。
    assert store.store_record(dict(candidate)) is None
    assert len(store.list_records()) == 1


# ---------------------------------------------------------------------------
# MEMORY.md 截断保护
# ---------------------------------------------------------------------------

def test_truncate_index_by_lines():
    text = "\n".join(f"- [mem{i}](mem{i}.md) - desc {i}" for i in range(250))
    out = truncate_index(text)
    assert "WARNING" in out
    # 前 200 行保留，尾部行被截掉。
    assert "mem0" in out
    assert "mem249" not in out
    assert len(out.splitlines()) <= 202  # 200 行内容 + 空行 + WARNING


def test_truncate_index_by_bytes():
    # 每行很长：行数不到 200 但 UTF-8 字节数超过 25KB。
    long_line = "- [x](x.md) - " + ("字" * 500)  # 约 1500+ 字节/行
    text = "\n".join([long_line] * 100)
    out = truncate_index(text)
    assert "WARNING" in out
    # 截断后字节数不超过上限太多（WARNING 文本本身占少量字节）。
    assert len(out.encode("utf-8")) <= 25_000 + 500


def test_truncate_index_passthrough():
    text = "- [a](a.md) - short catalog line"
    assert truncate_index(text) == text
    assert truncate_index("") == ""


def test_recall_block_truncates_huge_catalog(tmp_path):
    # 集成验证：205 条记忆 -> 索引超过 200 行 -> 注入块带 WARNING。
    mgr = MemoryManager(tmp_path)
    with exclusive_file_lock(mgr.store._lock_path):
        for i in range(205):
            mgr.store._write_record_unlocked(
                f"mem-{i:03d}", "project", f"第 {i} 条记忆的描述", f"第 {i} 条记忆正文内容。"
            )
        mgr.store._rebuild_index_unlocked()
    block = mgr.recall_block(
        [{"role": "user", "content": "mem-000 相关内容"}], ScriptedLLM([])
    )
    assert "WARNING" in block


# ---------------------------------------------------------------------------
# 陈旧记忆警告 + 策略行
# ---------------------------------------------------------------------------

def test_stale_warning_for_old_memory_mtime_fallback(tmp_path):
    # 旧格式记忆文件没有 updated 字段 -> 回退用文件 mtime 判断陈旧度。
    mgr = MemoryManager(tmp_path)
    mgr.store.memory_dir.mkdir(parents=True, exist_ok=True)
    old_file = mgr.store.memory_dir / "old-fact.md"
    old_file.write_text(
        "---\nname: old-fact\ntype: project\ndescription: 旧的发布流程\n---\n\n旧流程内容。\n",
        encoding="utf-8",
    )
    old_ts = time.time() - 40 * 86400  # 40 天前
    os.utime(old_file, (old_ts, old_ts))
    mgr.store.rebuild_index()

    block = mgr.recall_block(
        [{"role": "user", "content": "发布流程是什么"}], ScriptedLLM([])
    )
    assert "天未更新" in block
    assert "核实" in block


def test_fresh_memory_has_no_stale_warning(tmp_path):
    mgr = MemoryManager(tmp_path)
    mgr.store.write_record("部署流程", "project", "生产环境发布步骤", "先跑测试再发版。")
    block = mgr.recall_block(
        [{"role": "user", "content": "生产环境怎么部署？"}], ScriptedLLM([])
    )
    assert "天未更新" not in block


def test_recall_block_includes_memory_policy(tmp_path):
    # 记忆策略行：每次加载记忆都注入写记忆行为规则。
    mgr = MemoryManager(tmp_path)
    mgr.store.write_record("部署流程", "project", "生产环境发布步骤", "先跑测试再发版。")
    block = mgr.recall_block(
        [{"role": "user", "content": "怎么部署？"}], ScriptedLLM([])
    )
    assert "Memory writing policy" in block


def test_recall_block_with_scripted_llm_uses_keywords(tmp_path):
    # ScriptedLLM must not spend turns on memory calls; recall degrades to keywords.
    store_dir = tmp_path
    mgr = MemoryManager(store_dir)
    mgr.store.write_record("部署流程", "project", "生产环境发布步骤", "先跑测试再发版。")
    llm = ScriptedLLM([LLMResponse(content="SHOULD NOT BE CONSUMED")])
    messages = [{"role": "user", "content": "生产环境怎么部署？"}]
    block = mgr.recall_block(messages, llm)
    assert "部署流程" in block
    assert "Memory catalog" in block
    # scripted turn untouched
    assert len(llm._turns) == 1


def test_recall_block_empty_when_no_memories(tmp_path):
    mgr = MemoryManager(tmp_path)
    llm = ScriptedLLM([])
    assert mgr.recall_block([{"role": "user", "content": "hi"}], llm) == ""


# ---------------------------------------------------------------------------
# write_memory_file 受限工具
# ---------------------------------------------------------------------------

def test_write_memory_file_tool_stores_and_dedupes(tmp_path):
    store = MemoryStore(tmp_path)
    tool = WriteMemoryFileTool(store)
    result = tool.execute(
        name="reply-language", type="user",
        description="User prefers Chinese replies",
        body="Always reply in Chinese by default.",
    )
    assert "Stored" in result
    records = store.list_records()
    assert len(records) == 1
    assert records[0]["name"] == "reply-language"
    # 索引随写入自动重建。
    assert "reply-language.md" in store.read_index()

    # 再次写入等价内容：工具内乐观锁判定重复 -> 跳过，不新增文件。
    result2 = tool.execute(
        name="reply-language", type="user",
        description="User prefers Chinese replies",
        body="Always reply in Chinese by default.",
    )
    assert "Skipped" in result2
    assert len(store.list_records()) == 1


def test_write_memory_file_tool_rejects_invalid_records(tmp_path):
    store = MemoryStore(tmp_path)
    tool = WriteMemoryFileTool(store)
    # type 不在合法枚举内 -> 返回错误文本，不落盘。
    result = tool.execute(name="bad", type="weird", description="d", body="b")
    assert "Error" in result
    assert store.list_records() == []
    # 缺字段同样拒绝。
    result2 = tool.execute(name="bad2", type="user", description="", body="b")
    assert "Error" in result2
    assert store.list_records() == []


# ---------------------------------------------------------------------------
# forked agent 提取
# ---------------------------------------------------------------------------

class _ForkedExtractLLM:
    """模拟 forked 提取 agent 的模型：第 1 轮调 write_memory_file，第 2 轮纯文本收尾。"""

    scripted = False  # 非脚本模型：允许触发记忆提取

    def __init__(self, record_args=None):
        self.calls = 0
        self.record_args = record_args or {
            "name": "reply-language",
            "type": "user",
            "description": "User prefers Chinese replies",
            "body": "Always reply in Chinese by default.",
        }

    def chat(self, messages, tools=None, on_token=None):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(content="", tool_calls=[ToolCall(
                id="call_extract_1",
                name="write_memory_file",
                arguments=dict(self.record_args),
            )])
        # 工具结果回填后，模型返回纯文本结束自己的 Agent Loop。
        return LLMResponse(content="已写入 1 条记忆：reply-language。")


class _NoopExtractLLM:
    """模拟 forked agent 判断没有值得提取的信息：直接纯文本回复，不调工具。"""

    scripted = False

    def chat(self, messages, tools=None, on_token=None):
        return LLMResponse(content="NO_NEW_MEMORY")


def test_extract_persists_durable_records(tmp_path):
    """同步提取入口（测试用）：forked agent 调 write_memory_file 后记忆落盘。"""
    mgr = MemoryManager(tmp_path)
    llm = _ForkedExtractLLM()
    messages = [
        {"role": "user", "content": "以后都用中文回复我"},
        {"role": "assistant", "content": "好的，我会用中文回复。"},
    ]
    stored = mgr.extract(messages, llm)
    assert stored == 1
    records = mgr.store.list_records()
    assert len(records) == 1
    assert records[0]["name"] == "reply-language"
    assert records[0]["type"] == "user"
    assert records[0]["updated"]  # 新格式带 updated 日期
    assert "reply-language.md" in mgr.store.read_index()
    # forked agent 两轮：工具调用 + 纯文本收尾。
    assert llm.calls == 2


def test_extract_async_fire_and_forget(tmp_path):
    """异步提取入口：立即返回，后台守护线程完成落盘。"""
    mgr = MemoryManager(tmp_path)
    llm = _ForkedExtractLLM()
    messages = [
        {"role": "user", "content": "以后都用中文回复我"},
        {"role": "assistant", "content": "好的。"},
    ]
    mgr.extract_async(messages, llm)  # fire-and-forget，不阻塞
    # 轮询等待后台线程写盘（最多 15 秒）。
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and not mgr.store.list_records():
        time.sleep(0.05)
    records = mgr.store.list_records()
    assert len(records) == 1
    assert records[0]["name"] == "reply-language"
    # 再给后台线程一点时间收尾（trace/consolidate 判断），避免与 tmp 清理竞争。
    time.sleep(0.3)


def test_extract_noop_when_nothing_durable(tmp_path):
    mgr = MemoryManager(tmp_path)
    stored = mgr.extract(
        [{"role": "user", "content": "今天天气不错"}],
        _NoopExtractLLM(),
    )
    assert stored == 0
    assert mgr.store.list_records() == []


def test_extract_skipped_for_scripted_llm(tmp_path):
    mgr = MemoryManager(tmp_path)
    llm = ScriptedLLM([LLMResponse(content="SHOULD NOT BE CONSUMED")])
    messages = [
        {"role": "user", "content": "Remember I prefer Chinese replies."},
        {"role": "assistant", "content": "Got it."},
    ]
    # 同步入口：脚本模型直接跳过。
    assert mgr.extract(messages, llm) == 0
    assert mgr.store.list_records() == []
    assert len(llm._turns) == 1
    # 异步入口同样跳过：不应启动后台提取。
    mgr.extract_async(messages, llm)
    time.sleep(0.3)
    assert mgr.store.list_records() == []
    assert len(llm._turns) == 1


def test_disabled_manager_is_inert(tmp_path):
    mgr = MemoryManager(tmp_path, enabled=False)
    assert mgr.recall_block([{"role": "user", "content": "x"}], ScriptedLLM([])) == ""
    assert mgr.extract([], ScriptedLLM([])) == 0
