"""Manual token counting demo.

Run:
    python tests/token_count_demo.py
    python tests/token_count_demo.py orbit/agent.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from orbit.context import _fallback_token_count, _get_tokenizer, count_text_tokens, estimate_tokens


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("orbit/context.py")
    target = target if target.is_absolute() else project_root / target
    target = target.resolve()

    content = target.read_text(encoding="utf-8")
    messages = [{"role": "user", "content": content}]
    tokenizer = _get_tokenizer()
    tokenizer_name = getattr(tokenizer, "name", type(tokenizer).__name__) if tokenizer else "fallback"

    print(f"file: {target}")
    print(f"chars: {len(content)}")
    print(f"tokenizer: {tokenizer_name}")
    print(f"count_text_tokens(content): {count_text_tokens(content)}")
    print(f"estimate_tokens(messages): {estimate_tokens(messages)}")
    print(f"fallback_token_count(content): {_fallback_token_count(content)}")
    print()

    sample = "hello world 你好世界"
    print(f"sample: {sample}")
    print(f"sample count_text_tokens: {count_text_tokens(sample)}")
    print(f"sample fallback_token_count: {_fallback_token_count(sample)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
