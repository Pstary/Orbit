"""Session persistence - save and resume conversations.

Claude Code maintains session state via QueryEngine (1295 lines).
Orbit distills this to: JSON dump of messages + model config.

存在session里的文件的示例
{
  "id": "session_20260826_153000_ab12cd34",
  "model": "doubao-seed-1-6",
  "saved_at": "2026-08-26 15:30:00",
  "messages": [
    {
      "role": "user",
      "content": "帮我看一下这个文件"
    },
    {
      "role": "assistant",
      "content": "我先读取文件..."
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_xxx",
          "type": "function",
          "function": {
            "name": "read_file",
            "arguments": "{\"file_path\":\"README.md\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_xxx",
      "content": "README.md的文件内容..."
    }
  ]
}
"""

import json
import re
import time
import uuid
from pathlib import Path

SESSIONS_DIR = Path.home() / ".orbit" / "sessions"
_SAFE_SESSION_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_SESSION_ID_LEN = 100  # keep filenames comfortably under the OS limit

# 把用户传入的session_id清洗成安全文件名，避免空值、非法字符、超长名称和路径穿越。
def _normalize_session_id(session_id: str | None) -> str:
    if not session_id:
        return _new_session_id()

    name = session_id.strip().replace("\\", "/").split("/")[-1]
    name = _SAFE_SESSION_RE.sub("-", name).strip(".-_")
    # 如果名字太长就截断到最大长度，再清理一次首尾符号。
    if len(name) > _MAX_SESSION_ID_LEN:
        name = name[:_MAX_SESSION_ID_LEN].strip(".-_")
    return name or _new_session_id()


def _new_session_id() -> str:
    return f"session_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

# 根据 session_id 生成最终的session文件路径，并做一次安全校验。
def _session_path(session_id: str) -> Path:
    path = (SESSIONS_DIR / f"{_normalize_session_id(session_id)}.json").resolve()
    # 调用 resolve() 转成绝对路径，确保路径是安全的。
    root = SESSIONS_DIR.resolve()
    # 最终文件必须直接位于SESSIONS_DIR目录下。
    # 如果path.parent不是session根目录，就说明可能发生了路径逃逸，比如试图构造到别的目录。
    if root != path.parent:
        raise ValueError("Invalid session id")
    return path


def save_session(messages: list[dict], model: str, session_id: str | None = None) -> str:
    """Save conversation to disk. Returns the session ID."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    session_id = _normalize_session_id(session_id)

    data = {
        "id": session_id,
        "model": model,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "messages": messages,
    }

    path = _session_path(session_id)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return session_id


def load_session(session_id: str) -> tuple[list[dict], str] | None:
    """Load a saved session. Returns (messages, model) or None."""
    path = _session_path(session_id)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["messages"], data["model"]
    except (json.JSONDecodeError, KeyError, OSError):
        # a corrupt or truncated session file shouldn't crash resume
        return None


def list_sessions() -> list[dict]:
    """List available sessions, newest first."""
    if not SESSIONS_DIR.exists():
        return []

    sessions = []
    for f in sorted(SESSIONS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            # grab first user message as preview
            preview = ""
            for m in data.get("messages", []):
                if m.get("role") == "user" and m.get("content"):
                    preview = m["content"][:80]
                    break
            sessions.append({
                "id": data.get("id", f.stem),
                "model": data.get("model", "?"),
                "saved_at": data.get("saved_at", "?"),
                "preview": preview,
            })
        except (json.JSONDecodeError, KeyError):
            continue

    return sessions[:20]  # cap at 20
