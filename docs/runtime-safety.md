# Runtime guarantees

- File tool paths are resolved relative to the harness workspace, including
  omitted search paths. Approval checks and execution use the same absolute path.
- Directory searches check each result against the workspace boundary and the
  sensitive-file policy. A symlink to an external file is excluded.
- Tool batches containing mutations or delegation run in model order. Only
  batches of tools marked read-only run concurrently. Built-in file edits and
  writes also share a process-wide lock to prevent lost updates between agents.
- Plan mode rejects mutations before approval and tool allowlists are considered.

## Tool deadlines

In-process Python tools have a **soft deadline**. Built-in file writes check the
deadline before writing. A running Python thread cannot safely be killed; if a
tool does not cooperate, the harness waits for it to finish before returning a
timeout or interruption. It does not leave the worker running after reporting a
timeout. Side effects completed before return are not rolled back. Mutating tools
are not automatically retried, and timeout errors are never retried.

This means a custom tool that never returns can still block shutdown. Such tools
must implement their own bounded I/O or cancellable execution. The deadline is
not a hard process-isolation boundary. Shell tools retain their sandbox runner's
process timeout behavior.

## Interactive interruption

Interactive turns run off the terminal input thread. Pressing Ctrl+C requests
cooperative cancellation through the active agent runtime. Shell commands run
in a dedicated process group; cancellation terminates the command tree and
waits for it to exit before the prompt is shown again. Completed tool results
are retained, while pending calls receive an `[interrupted]` result so the next
user message can safely steer and continue the same conversation.

In-process tools observe cancellation through `check_deadline()`. A custom tool
that does not call this helper cannot be killed safely, so Orbit waits for it to
finish before returning control and allowing another turn to start.

MCP stdout is consumed by a dedicated reader and buffered behind a deadline-aware
queue on every platform. Silence, partial lines and partial frame bodies all
respect the request deadline. Failed initialization cleans up the server process.

## Memory shutdown

Harness shutdown waits up to 30 seconds for background memory extraction across
registered agents before saving the final trace. CLI, evaluation and library
callers share this lifecycle. If extraction is still pending, the configured
notification callback reports that the current turn's memory may be lost, and
the trace records a timeout warning.
This is a bounded grace period, not a durable background job queue.

These checks do not turn the local shell backend or third-party MCP servers into
an operating-system sandbox.
