"""skills加载工具。"""

from pathlib import Path
from typing import ClassVar

from ..skill_registry import SkillRegistry, default_skills_dir
from .base import Tool


class LoadSkillTool(Tool):
    name = "load_skill"
    read_only = True
    description = "Load the full SKILL.md instructions for a workspace skill by name."
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Skill name from the skills catalog.",
            },
        },
        "required": ["name"],
    }

    def __init__(self, skills_dir: Path | str | None = None):
        # 未显式指定skills目录时，使用统一默认发现规则。
        self.registry = SkillRegistry(skills_dir or default_skills_dir())

    def catalog(self) -> str:
        """返回给系统提示词的轻量skills目录。"""
        # 每次读取目录前重新扫描，保证REPL运行中新增skills也能被/slash命令看到。
        self.registry.scan()
        return self.registry.catalog()

    def list_skills(self) -> list[dict[str, str]]:
        """返回CLI展示用的skills清单。"""
        # /skills展示实时目录，避免用户新增SKILL.md后必须重启进程。
        self.registry.scan()
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "path": str(skill.path),
            }
            for skill in self.registry.skills.values()
        ]

    def execute(self, name: str) -> str:
        # load_skill调用前刷新注册表，保证模型能加载最新落盘的技能文件。
        self.registry.scan()
        return self.registry.load(name)
