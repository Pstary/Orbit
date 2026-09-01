"""write_memory_file：持久记忆写入工具（仅供记忆提取 forked agent 使用）。

设计要点（对应 Claude Code 的持久记忆机制）：
- 这个工具【不】注册进 get_default_tools()，主 Agent 拿不到它；只有后台
  forked 提取 agent 的工具集里才有 read_file + write_memory_file 两个工具，
  因此 forked agent 不能执行 shell、不能改项目代码（受限权限）。
- 每次调用写入一条记忆：工具内部做参数校验 -> 交给 MemoryStore.store_record()
  在独占锁内"重读现有记忆 -> 乐观锁去重 -> 原子写文件 -> 重建 MEMORY.md 索引"。
- 返回值是给 forked agent 看的文本：成功写入 / 因重复被跳过 / 参数错误。
"""

from typing import ClassVar

from .base import Tool


class WriteMemoryFileTool(Tool):
    name = "write_memory_file"
    read_only = False
    description = (
        "Write ONE durable memory record into the persistent memory store. "
        "Call this once per extracted memory. The record is validated, "
        "deduplicated against existing memories (duplicates are skipped), "
        "written atomically, and the MEMORY.md catalog is rebuilt automatically. "
        "Store only cross-session knowledge: durable user preferences, repeated "
        "feedback, stable project facts, or explicitly requested references."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": (
                    "Short memory name, also used as the filename stem "
                    "(e.g. 'reply-language' or '部署流程'). Must be unique."
                ),
            },
            "type": {
                "type": "string",
                "enum": ["user", "feedback", "project", "reference"],
                "description": (
                    "user = durable user preference; "
                    "feedback = repeated user feedback/correction; "
                    "project = stable project fact/convention; "
                    "reference = external reference the user asked to remember."
                ),
            },
            "description": {
                "type": "string",
                "description": "One-line summary used by the catalog for future recall searches.",
            },
            "body": {
                "type": "string",
                "description": "Full, self-contained memory content that remains useful in future sessions.",
            },
        },
        "required": ["name", "type", "description", "body"],
    }

    def __init__(self, store):
        # store 是 orbit.memory.MemoryStore 实例；构造时注入，避免工具反向依赖记忆模块。
        self.store = store

    def execute(self, name: str = "", type: str = "", description: str = "", body: str = "") -> str:
        # 组装候选记录。scope 固定为 persistent：经此工具写入的都是跨会话持久记忆，
        # should_store() 会据此过滤掉临时内容（含 this session/本次会话 等标记）。
        candidate = {
            "name": name,
            "type": type,
            "description": description,
            "body": body,
            "scope": "persistent",
        }
        # 第一层校验：必填字段 + type 合法枚举；非法时返回错误文本让 forked agent 修正重试。
        validated = self.store.validate_record(candidate, require_scope=True)
        if validated is None:
            return (
                "Error: invalid memory record. All of name/type/description/body are "
                "required, and type must be one of: user, feedback, project, reference."
            )
        try:
            # 第二层：乐观锁写入。store_record() 在独占锁内重读磁盘现有记忆，
            # 若主 Agent 或并发提取已经写入等价内容（同名/同描述/同正文），
            # 直接返回 None 跳过，防止异步竞态产生重复记忆。
            path = self.store.store_record(validated)
        except Exception as exc:  # noqa: BLE001 - 工具失败返回文本，不向 agent 抛异常
            return f"Error writing memory: {exc}"
        if path is None:
            return (
                f"Skipped: an equivalent or temporary-sounding memory already exists "
                f"for '{name}'. Do not write it again."
            )
        return f"Stored memory '{name}' -> {path.name}"
