---
name: Orbit离线演示脚本特性
type: project
description: Orbit项目内置离线演示脚本orbit/demo.py的核心功能与运行特性
---

该脚本为无需LLM API密钥的离线演示入口：使用ScriptedLLM执行预设脚本任务，可完整复现真实模型下的「思考-工具调用-观测」Agent循环流程，适用于功能演示、截图或Agent循环冒烟测试；运行时默认关闭记忆功能，自动创建系统临时工作目录，运行结束后会输出临时工作区路径与Trace保存路径。
