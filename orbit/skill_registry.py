"""skills发现与加载。

skills是工作区内的指令包，默认目录结构如下：

    skills/<name>/SKILL.md

系统提示词只注入轻量目录，完整SKILL.md通过load_skill按需读取，
避免普通对话为所有skills支付token成本。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    content: str
    path: Path


class SkillRegistry:
    """扫描workspace内的SKILL.md，并提供按名称加载能力。"""

    def __init__(self, skills_dir: Path | str):
        # skills_dir由CLI或默认规则传入，注册表只负责扫描该目录。
        self.skills_dir = Path(skills_dir).expanduser()
        self.skills: dict[str, Skill] = {}
        self.scan()

    @staticmethod
    def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
        """解析独立`---`包裹的frontmatter。

        优先使用PyYAML（如果环境已有），没有该依赖时退化到小型解析器，
        覆盖技能清单需要的`name`和`description`字段。
        """

        lines = text.splitlines(keepends=True)
        # frontmatter必须由单独一行`---`开始，避免误解析正文里的分隔符。
        if not lines or lines[0].rstrip("\r\n") != "---":
            return {}, text

        closing_index = next(
            (index for index, line in enumerate(lines[1:], start=1) if line.rstrip("\r\n") == "---"),
            None,
        )
        if closing_index is None:
            return {}, text

        frontmatter = "".join(lines[1:closing_index])
        body = "".join(lines[closing_index + 1 :]).strip()
        return _parse_metadata(frontmatter), body

    def scan(self) -> None:
        self.skills.clear()
        if not self.skills_dir.exists():
            return

        skills_root = self.skills_dir.resolve()
        for manifest in sorted(self.skills_dir.glob("*/SKILL.md")):
            if not manifest.is_file():
                continue
            resolved = manifest.resolve()
            # 跳过指向skills目录外部的软链接，避免load_skill绕过工作区治理。
            if not resolved.is_relative_to(skills_root):
                continue

            content = manifest.read_text(encoding="utf-8")
            metadata, body = self.parse_frontmatter(content)
            raw_name = metadata.get("name")
            name = raw_name.strip() if isinstance(raw_name, str) else ""
            name = name or manifest.parent.name
            raw_description = metadata.get("description")
            description = raw_description.strip() if isinstance(raw_description, str) else ""
            description = description or _description_from_body(body)
            # 系统提示词里的目录必须保持单行，避免技能摘要撑大上下文。
            description = " ".join(description.split())
            self.skills[name] = Skill(
                name=name,
                description=description,
                content=content,
                path=resolved,
            )

    def catalog(self) -> str:
        if not self.skills:
            return "(no skills found)"
        return "\n".join(
            f"- {skill.name}: {skill.description}"
            for skill in self.skills.values()
        )

    def load(self, name: str) -> str:
        skill = self.skills.get(name)
        if skill is not None:
            return skill.content
        available = ", ".join(self.skills) or "none"
        return f"Error: Unknown skill '{name}'. Available: {available}"


def _parse_metadata(frontmatter: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        metadata = yaml.safe_load(frontmatter) or {}
        return metadata if isinstance(metadata, dict) else {}
    except Exception:  # noqa: BLE001
        # PyYAML不存在或解析失败时，退化解析常用的name/description写法。
        return _parse_simple_yaml(frontmatter)


def _parse_simple_yaml(frontmatter: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    lines = frontmatter.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip() or line.lstrip().startswith("#") or line.startswith((" ", "\t")):
            index += 1
            continue
        if ":" not in line:
            index += 1
            continue

        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if value in {"|", ">"}:
            block: list[str] = []
            index += 1
            while index < len(lines):
                candidate = lines[index]
                if candidate and not candidate.startswith((" ", "\t")) and ":" in candidate:
                    break
                block.append(candidate.strip())
                index += 1
            metadata[key] = "\n".join(block).strip() + ("\n" if block else "")
            continue

        metadata[key] = value.strip("\"'")
        index += 1
    return metadata


def _description_from_body(body: str) -> str:
    for line in body.splitlines():
        text = " ".join(line.lstrip("# ").split())
        if text:
            return text
    return ""


def default_skills_dir(base_dir: Path | str | None = None) -> Path:
    """返回默认技能目录。"""

    root = Path(base_dir).expanduser() if base_dir is not None else Path.cwd()
    candidates = [
        root / "skills",
        root / "orbit" / "skills",
    ]
    for candidate in candidates:
        # 优先使用项目根目录skills；不存在时兼容包内orbit/skills。
        if candidate.exists():
            return candidate
    return candidates[0]
