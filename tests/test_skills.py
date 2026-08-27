from io import StringIO

from rich.console import Console

from orbit import cli as cli_module
from orbit.agent import Agent
from orbit.llm import LLM, LLMResponse, ScriptedLLM, ToolCall
from orbit.skill_registry import SkillRegistry, default_skills_dir
from orbit.tools import get_default_tools


def test_skill_catalog_stays_small_and_load_returns_full_manifest(tmp_path):
    skill_dir = tmp_path / "skills" / "code-review"
    skill_dir.mkdir(parents=True)
    manifest = """---
name: code-review
description: |
  Review code for bugs,
  regressions, and missing tests.
---

# Code Review

UNIQUE_FULL_INSTRUCTION
"""
    (skill_dir / "SKILL.md").write_text(manifest, encoding="utf-8")

    registry = SkillRegistry(tmp_path / "skills")

    assert registry.catalog() == "- code-review: Review code for bugs, regressions, and missing tests."
    assert registry.load("code-review") == manifest

    tools = get_default_tools(include_mcp=False, skills_dir=str(tmp_path / "skills"))
    load_skill = next(tool for tool in tools if tool.name == "load_skill")
    assert load_skill.execute("code-review") == manifest

    agent = Agent(llm=LLM.__new__(LLM), tools=tools)
    assert "code-review" in agent._system
    assert "UNIQUE_FULL_INSTRUCTION" not in agent._system


def test_skill_loader_reads_utf8_and_falls_back_to_body_description(tmp_path):
    manifest = """---
name: chinese-skill
description: 处理中文内容
---

# 中文技能
"""
    fallback = """---
name:
description:
---
# Body description
"""
    (tmp_path / "skills" / "chinese-skill").mkdir(parents=True)
    (tmp_path / "skills" / "chinese-skill" / "SKILL.md").write_text(manifest, encoding="utf-8")
    (tmp_path / "skills" / "fallback-skill").mkdir()
    (tmp_path / "skills" / "fallback-skill" / "SKILL.md").write_text(fallback, encoding="utf-8")

    registry = SkillRegistry(tmp_path / "skills")

    assert registry.skills["chinese-skill"].description == "处理中文内容"
    assert registry.load("chinese-skill") == manifest
    assert registry.skills["fallback-skill"].description == "Body description"


def test_skill_frontmatter_requires_standalone_delimiters():
    invalid_opening = "---not frontmatter\n---\n# Body"
    block_scalar = """---
name: demo
description: |
  before
  ---
  after
---
# Body
"""

    assert SkillRegistry.parse_frontmatter(invalid_opening) == ({}, invalid_opening)
    metadata, body = SkillRegistry.parse_frontmatter(block_scalar)
    assert metadata["description"] == "before\n---\nafter\n"
    assert body == "# Body"


def test_skill_loader_skips_symlinks_outside_skills_dir(tmp_path):
    outside = tmp_path / "outside-skill.md"
    outside.write_text("# External skill\n\nDO_NOT_LOAD", encoding="utf-8")
    linked_dir = tmp_path / "skills" / "linked-skill"
    linked_dir.mkdir(parents=True)
    (linked_dir / "SKILL.md").symlink_to(outside)

    registry = SkillRegistry(tmp_path / "skills")

    assert "linked-skill" not in registry.skills
    assert "linked-skill" not in registry.catalog()


def test_agent_reports_loaded_skill_through_tool_result_callback(tmp_path):
    skill_dir = tmp_path / "skills" / "frontend-design"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: frontend-design\ndescription: UI guidance\n---\n# Body\n",
        encoding="utf-8",
    )
    tools = get_default_tools(include_mcp=False, skills_dir=str(tmp_path / "skills"))
    llm = ScriptedLLM([
        LLMResponse(
            content="loading",
            tool_calls=[
                ToolCall(
                    id="skill-1",
                    name="load_skill",
                    arguments={"name": "frontend-design"},
                )
            ],
        ),
        LLMResponse(content="done"),
    ])
    seen = []
    agent = Agent(llm=llm, tools=tools, max_rounds=3)

    assert agent.chat("use the frontend skill", on_tool_result=lambda *args: seen.append(args)) == "done"
    assert seen[0][0] == "load_skill"
    assert seen[0][1] == {"name": "frontend-design"}
    assert "# Body" in seen[0][2]


def test_cli_can_list_skills_and_render_load_status(tmp_path, monkeypatch):
    skill_dir = tmp_path / "skills" / "frontend-design"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: frontend-design\ndescription: UI guidance\n---\n# Body\n",
        encoding="utf-8",
    )
    tools = get_default_tools(include_mcp=False, skills_dir=str(tmp_path / "skills"))
    agent = Agent(llm=LLM.__new__(LLM), tools=tools)
    output = StringIO()
    monkeypatch.setattr(cli_module, "console", Console(file=output, force_terminal=False, width=120))

    cli_module._show_skills(agent)
    cli_module._print_tool_start("load_skill", {"name": "frontend-design"})
    cli_module._print_tool_result("load_skill", {"name": "frontend-design"}, "# Body")

    text = output.getvalue()
    assert "Available skills:" in text
    assert "1. frontend-design：UI guidance" in text
    assert "UI guidance" in text
    assert "SKILL.md" not in text
    assert "Skill(frontend-design)" in text
    assert "Successfully loaded skill" in text


def test_write_file_brief_hides_content_body():
    summary = cli_module._brief({
        "file_path": "login.html",
        "content": "background\nbody { color: red; }",
    })

    assert "file_path='login.html'" in summary
    assert "content='<31 chars>'" in summary
    assert "background" not in summary


def test_tool_result_prints_write_file_status(monkeypatch):
    output = StringIO()
    monkeypatch.setattr(cli_module, "console", Console(file=output, force_terminal=False, width=120))

    cli_module._print_tool_result("write_file", {}, "Wrote 42 lines to login.html")

    assert "Wrote 42 lines to login.html" in output.getvalue()


def test_tool_result_prints_non_skill_errors(monkeypatch):
    output = StringIO()
    monkeypatch.setattr(cli_module, "console", Console(file=output, force_terminal=False, width=120))

    cli_module._print_tool_result(
        "write_file",
        {},
        "Error: bad arguments for write_file: missing a required argument: 'file_path'",
    )

    text = output.getvalue()
    assert "bad arguments for write_file" in text
    assert "file_path" in text


def test_default_skill_dir_falls_back_to_package_skills(tmp_path, monkeypatch):
    skill_dir = tmp_path / "orbit" / "skills" / "frontend_design"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: frontend-design\ndescription: UI guidance\n---\n# Body\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert default_skills_dir() == tmp_path / "orbit" / "skills"

    tools = get_default_tools(include_mcp=False)
    load_skill = next(tool for tool in tools if tool.name == "load_skill")
    assert "frontend-design" in load_skill.catalog()
    assert "# Body" in load_skill.execute("frontend-design")


def test_load_skill_tool_rescans_before_listing_and_loading(tmp_path):
    tools = get_default_tools(include_mcp=False, skills_dir=str(tmp_path / "skills"))
    load_skill = next(tool for tool in tools if tool.name == "load_skill")
    assert load_skill.list_skills() == []

    skill_dir = tmp_path / "skills" / "frontend-design"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: frontend-design\ndescription: UI guidance\n---\n# Body\n",
        encoding="utf-8",
    )

    assert load_skill.list_skills()[0]["name"] == "frontend-design"
    assert "# Body" in load_skill.execute("frontend-design")


def test_tool_call_round_streams_intermediate_text(tmp_path):
    skill_dir = tmp_path / "skills" / "frontend-design"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: frontend-design\ndescription: UI guidance\n---\n# Body\n",
        encoding="utf-8",
    )
    tools = get_default_tools(include_mcp=False, skills_dir=str(tmp_path / "skills"))
    llm = ScriptedLLM([
        LLMResponse(
            content="background",
            tool_calls=[
                ToolCall(
                    id="skill-1",
                    name="load_skill",
                    arguments={"name": "frontend-design"},
                )
            ],
        ),
        LLMResponse(content="done"),
    ])
    streamed = []
    agent = Agent(llm=llm, tools=tools, max_rounds=3)

    assert agent.chat("use the frontend skill", on_token=streamed.append) == "done"
    assert streamed == ["background", "done"]
