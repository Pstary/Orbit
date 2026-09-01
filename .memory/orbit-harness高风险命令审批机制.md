---
name: Orbit harness高风险命令审批机制
type: project
description: Orbit项目harness模块的命令安全审批规则，高风险命令需人工审批后方可执行
updated: 2026-09-01
---

Orbit项目内置harness命令执行安全模块，对命令执行进行风险管控：
1. 被标记为[HIGH]级别的高风险命令会被自动拦截，无法直接运行，必须经过人工审批后才可执行；
2. 明确属于高风险类别的操作包括：PowerShell文件删除、递归删除、强制删除。
