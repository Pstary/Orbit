# Harness-owned runtime

`Agent` owns only the LLM connection and a reference to its harness. Its
`respond()` method makes one model call. `chat()` delegates to
`OrbitHarness.run_chat()`; it does not contain an execution loop.

```text
CLI / evaluation / Python caller
  -> OrbitHarness.run_chat(agent, input)
       -> prepare conversation, compress context, recall memory
       -> repeat within the round limit:
            -> Agent.respond(messages, schemas) -> LLM.chat()
            -> append response
            -> validate, approve and schedule tools
            -> append results and compress context
       -> finish conversation and schedule memory extraction
  -> OrbitHarness.close()
       -> wait for memory extraction (bounded)
       -> close shared tool clients once
       -> shutdown hooks and final trace
```

## Ownership

- `harness/core.py`: runtime entry, hooks, permission checks, sandbox, tool
  deadlines, retries, tracing and shutdown.
- `harness/runtime.py`: the conversation loop, per-agent `AgentRuntime` records,
  messages, prompts, context, tool discovery and scheduling, sub-agent delegation,
  session save/resume, memory creation/recall/extraction/status and cleanup.
- `memory.py`: the memory storage and extraction service used by the harness.
  The extraction agent has its own harness and disables recursive memory.
- `agent.py`: model-facing `respond()` and compatibility delegates.
- `llm.py`: provider calls, streaming, response parsing and provider retries.
- `cli.py`: argument parsing, terminal input/output and approval UI. It requests
  operations from the harness and calls `close()` in one `finally` block.

Each agent registered with a harness has separate history, context and memory
configuration. Sub-agents use the same workspace and permissions, disable memory,
and release their conversation context after finishing. Delegation tool bindings
are copied per agent to prevent one agent rebinding another's tool.

Existing `agent.messages`, `agent.tools`, `agent.context`, and `agent.memory`
remain compatibility views of harness-owned state. They do not create another
copy of that state. `agent.chat()` and `harness.run_chat()` execute the same
lifecycle exactly once. Calls after harness shutdown are rejected.

## Library use

```python
from orbit import LLM
from orbit.harness import HarnessConfig, OrbitHarness

llm = LLM(model="your-model", api_key="your-key")
with OrbitHarness(HarnessConfig()) as harness:
    agent = harness.create_agent(llm)
    answer = harness.run_chat(agent, "Explain this repository")
    session_id = harness.save_session(agent)
```

The context manager also closes the runtime on exceptions. The same cleanup is
used by CLI and evaluation runs. Memory shutdown defaults to a 30-second shared
grace period across registered agents, configurable through
`HarnessConfig.memory_shutdown_timeout_seconds`. It remains a best-effort grace
period rather than a durable background queue. In-process tool deadlines remain
soft deadlines as described in `runtime-safety.md`.
