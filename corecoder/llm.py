"""LLM provider layer - thin wrapper over OpenAI-compatible APIs.

Since most providers (DeepSeek, Qwen, Kimi, GLM, Ollama, etc.) expose an
OpenAI-compatible endpoint, we just use the openai SDK directly.  Switch
provider by changing OPENAI_BASE_URL + OPENAI_API_KEY. That's it.

For providers that are NOT OpenAI-compatible (AWS Bedrock, Google Vertex,
etc.), use the LiteLLM backend which routes to 100+ providers through a
single unified interface. Set CORECODER_PROVIDER=litellm.
"""

import json
import time
from dataclasses import dataclass, field

from openai import APIConnectionError, APIError, APITimeoutError, BadRequestError, OpenAI, RateLimitError


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

# LLMResponse 是CoreCoder内部的“模型响应DTO”。它把外部SDK返回、流式chunk、工具调用和token统计统一封装起来，让 Agent 层只处理稳定的业务语义：模型说了什么、要调什么工具、消耗了多少token。
@dataclass
class LLMResponse:
    content: str = ""
    # 定义了模型回复里的工具调用列表，并用default_factory=list避免多个回复对象共享同一个可变列表
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def message(self) -> dict:
        """Convert to OpenAI message format for appending to history."""
        msg: dict = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        # tc.arguments 在程序内部是Python字典，但OpenAI协议里的 function.arguments 要求是JSON字符串，所以要用 json.dumps() 转成字符串。
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]
        return msg


# pricing per million tokens: (input, output)
# sources: openai.com/api/pricing, api-docs.deepseek.com, platform.claude.com,
#          platform.moonshot.ai, alibabacloud.com/help/en/model-studio
# 这里定义了不同模型的token价格，单位是美元/百万token。
# 注意，这里只是参考值，实际价格可能会有变化。
_PRICING = {
    # OpenAI - current flagships
    "gpt-5.5": (5, 30),
    "gpt-5.4": (2.5, 15),
    "gpt-5.4-mini": (0.75, 4.5),
    "gpt-5.4-nano": (0.2, 1.25),
    "o4-mini": (1.1, 4.4),
    # OpenAI - previous gen (still widely used)
    "gpt-4.1": (2, 8),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.1, 0.4),
    "gpt-4o": (2.5, 10),
    "gpt-4o-mini": (0.15, 0.6),
    # DeepSeek
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    # Anthropic Claude
    "claude-opus-4-6": (5, 25),
    "claude-sonnet-4-6": (3, 15),
    "claude-haiku-4-5": (1, 5),
    # Alibaba Qwen
    "qwen3-max": (0.78, 3.9),
    "qwen3-plus": (0.26, 0.78),
    "qwen-max": (0.78, 3.9),
    # Moonshot Kimi
    "kimi-k2.5": (0.6, 3),
}

# 用固定脚本模拟真实模型，让Agent主循环、工具调用、测试用例可以在没有API key、没有网络、结果可复现的情况下运行。
class ScriptedLLM:
    """Deterministic offline LLM for demos and smoke tests.

    Plays back a list of LLMResponse turns, one per chat() call, streaming
    each turn's content through on_token. Running out of turns is an error,
    not a silent hang, so a broken loop shows up immediately.
    """

    total_prompt_tokens = 0
    total_completion_tokens = 0

    def __init__(self, script: list[LLMResponse], model: str = "scripted-demo"):
        # 保存一份预设回复脚本，每次chat()按顺序取出一条，避免修改调用方传入的原始列表。
        self._turns = list(script)
        self.model = model

    def chat(self, messages, tools=None, on_token=None) -> LLMResponse:
        # 脚本耗尽说明测试流程超出了预期轮数，直接报错比静默返回更容易暴露主循环问题。
        if not self._turns:
            raise RuntimeError("ScriptedLLM ran out of turns")
        # 模拟真实LLM调用：每次chat()消费一条预设LLMResponse。
        resp = self._turns.pop(0)
        # 如果外层传了on_token回调，就把完整文本一次性推给它，模拟流式输出。
        if on_token and resp.content:
            on_token(resp.content)
        # 离线模型没有真实usage，这里用空格分词粗略累加completion token，供/tokens等展示使用。
        self.total_completion_tokens += len(resp.content.split())
        return resp

# 这是默认模型客户端，基于OpenAISDK，支持所有OpenAI-compatible接口
class LLM:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        **kwargs,
    ):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.extra = kwargs  # temperature, max_tokens, etc.
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    @property
    def estimated_cost(self) -> float | None:
        """Rough cost estimate in USD. Returns None if model not in pricing table."""
        pricing = _PRICING.get(self.model)
        if not pricing:
            return None
        input_rate, output_rate = pricing
        return (
            self.total_prompt_tokens * input_rate / 1_000_000
            + self.total_completion_tokens * output_rate / 1_000_000
        )

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_token=None,
    ) -> LLMResponse:
        """Send messages, stream back response, handle tool calls."""
        params: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            **self.extra,
        }
        if tools:
            params["tools"] = tools

        # stream_options is an OpenAI extension; fall back only when the provider
        # rejects the param (400 BadRequest), not on transient errors that
        # _call_with_retry already exhausted - otherwise we'd double the retries
        # 请求模型时开启stream流式输出，同时要求服务端在流结束时返回usage信息。
        params["stream_options"] = {"include_usage": True}
        # stream_options 的降级兼容逻辑，先尝试带usage统计请求，服务端不支持就删掉该参数再请求。
        try:
            stream = self._call_with_retry(params)
        except BadRequestError:
            params.pop("stream_options", None)
            stream = self._call_with_retry(params)

        # 用列表累积模型流式返回的正文片段，最后再join成完整content。
        content_parts: list[str] = []
        # 用index聚合流式返回的tool_call分片，同一个工具调用可能被拆成多个chunk返回。
        tc_map: dict[int, dict] = {}  # index -> {id,name,arguments_str}
        # 初始化输入token计数，默认0，等流最后的usage回来再覆盖。
        prompt_tok = 0
        # 初始化输出token计数，默认0，等流最后的usage回来再覆盖。
        completion_tok = 0

        # 遍历服务端流式返回的每一个chunk。
        for chunk in stream:
            # usage通常只在最后一个chunk里返回，用来统计token消耗。
            if chunk.usage:
                # 有些provider会返回null字段，这里用or 0避免后面int和None相加报错。
                prompt_tok = chunk.usage.prompt_tokens or 0
                # 记录模型输出消耗的completion token数量。
                completion_tok = chunk.usage.completion_tokens or 0

            # 如果当前chunk没有choices，说明它不包含正文或工具调用增量，直接跳过。
            if not chunk.choices:
                continue
            # 取第一个choice的delta，delta表示本次流式增量内容。
            delta = chunk.choices[0].delta

            # 如果delta里有正文内容，就把这段文本加入正文缓冲区。
            if delta.content:
                # 保存当前文本片段，后面拼成完整回复。
                content_parts.append(delta.content)
                # 如果外层传了on_token回调，就把当前文本片段实时推出去。
                if on_token:
                    # 触发流式输出回调，让CLI或UI可以边生成边展示。
                    on_token(delta.content)

            # 如果delta里有工具调用内容，就开始累积tool_call分片。
            if delta.tool_calls:
                # 一次chunk里可能包含多个tool_call增量，逐个处理。
                for tc_delta in delta.tool_calls:
                    # index标识这是第几个工具调用，用它把分片合并回同一个调用。
                    idx = tc_delta.index
                    # 第一次看到这个index时，先初始化一个临时聚合结构。
                    if idx not in tc_map:
                        tc_map[idx] = {"id": "", "name": "", "args": ""}
                    # 如果当前分片带了tool_call id，就记录下来。
                    if tc_delta.id:
                        tc_map[idx]["id"] = tc_delta.id
                    # 如果当前分片带了function字段，就继续解析函数名和参数。
                    if tc_delta.function:
                        # function name通常只在某个分片里出现，出现时保存。
                        if tc_delta.function.name:
                            tc_map[idx]["name"] = tc_delta.function.name
                        # function arguments是JSON字符串，流式返回时可能被拆碎，所以要追加拼接。
                        if tc_delta.function.arguments:
                            tc_map[idx]["args"] += tc_delta.function.arguments

        # parse accumulated tool calls
        parsed: list[ToolCall] = []
        for idx in sorted(tc_map):
            raw = tc_map[idx]
            try:
                args = json.loads(raw["args"])
            except (json.JSONDecodeError, KeyError):
                args = {}
            parsed.append(ToolCall(id=raw["id"], name=raw["name"], arguments=args))

        self.total_prompt_tokens += prompt_tok
        self.total_completion_tokens += completion_tok

        return LLMResponse(
            content="".join(content_parts),
            tool_calls=parsed,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
        )

    # 这个函数负责给模型API调用加指数退避重试，只重试限流、超时、网络错误和5xx服务端错误；参数错误、鉴权错误等4xx会直接失败。
    def _call_with_retry(self, params: dict, max_retries: int = 3):
        """Retry on transient errors with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return self.client.chat.completions.create(**params)
            except (RateLimitError, APITimeoutError, APIConnectionError):
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                time.sleep(wait)
            except APIError as e:
                # retry 5xx server errors but not 4xx; base APIError has no status_code so read it defensively
                status_code = getattr(e, "status_code", None)
                if status_code and status_code >= 500 and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

# 接入更杂的provider，比如Bedrock、VertexAI等。
class LiteLLM(LLM):
    # 定义LiteLLM后端类，继承LLM以复用统一的接口形态。
    """LLM backend via LiteLLM, supporting 100+ providers.

    Use this when your target provider is NOT OpenAI-compatible
    (AWS Bedrock, Google Vertex, Cohere, etc.) or when you want
    a single interface to switch between any provider by changing
    the model string.

    Set CORECODER_PROVIDER=litellm and use LiteLLM model strings
    like ``anthropic/claude-3-haiku``, ``bedrock/anthropic.claude-v2``,
    ``vertex_ai/gemini-pro``, etc.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        # 跳过LLM.__init__，因为这里不创建OpenAI客户端，而是后面直接调用litellm.completion。
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.extra = kwargs
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_token=None,
    ) -> LLMResponse:
        # 通过LiteLLM发送messages，并把流式响应整理成统一的LLMResponse。
        """Send messages via litellm, stream back response, handle tool calls."""
        # 组装LiteLLM请求参数。
        params: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            **self.extra,
        }
        # 如果Agent传入了工具schema，就把工具声明一起发给模型。
        if tools:
            params["tools"] = tools

        # 请求最后一个chunk携带usage统计；不支持的provider会被drop_params自动丢弃。
        params["stream_options"] = {"include_usage": True}
        # 使用带重试的LiteLLM调用拿到流式响应对象。
        stream = self._call_with_retry(params)

        # 用列表累积模型流式返回的正文片段，最后再join成完整content。
        content_parts: list[str] = []
        # 用index聚合流式返回的tool_call分片，同一个工具调用可能被拆成多个chunk返回。
        tc_map: dict[int, dict] = {}
        # 初始化输入token计数，默认0，等流最后的usage回来再覆盖。
        prompt_tok = 0
        # 初始化输出token计数，默认0，等流最后的usage回来再覆盖。
        completion_tok = 0

        # 遍历LiteLLM流式返回的每一个chunk。
        for chunk in stream:
            # 用getattr读取usage，避免不同provider返回对象没有usage字段时报错。
            usage = getattr(chunk, "usage", None)
            # 如果当前chunk带usage，就提取token统计。
            if usage:
                # 读取输入token数；字段不存在或为None时按0处理。
                prompt_tok = getattr(usage, "prompt_tokens", 0) or 0
                # 读取输出token数；字段不存在或为None时按0处理。
                completion_tok = getattr(usage, "completion_tokens", 0) or 0

            # 如果当前chunk没有choices，说明没有可解析的内容，直接进入下一个chunk。
            if not getattr(chunk, "choices", None):
                continue
            # 取第一个choice的delta，delta代表这次流式增量。
            delta = chunk.choices[0].delta

            # 如果delta里有正文内容，就处理文本输出。
            if getattr(delta, "content", None):
                # 把当前文本片段加入缓冲区。
                content_parts.append(delta.content)
                # 如果外层传了on_token回调，就实时通知外层。
                if on_token:
                    # 把当前文本片段推出去，让CLI或UI可以边生成边展示。
                    on_token(delta.content)

            # 如果delta里有工具调用内容，就处理tool_call增量。
            if getattr(delta, "tool_calls", None):
                # 一个chunk里可能包含多个工具调用分片，逐个处理。
                for tc_delta in delta.tool_calls:
                    # index表示第几个工具调用，用它把多个分片聚合到同一个调用里。
                    idx = tc_delta.index
                    # 第一次遇到该index时，初始化临时聚合结构。
                    if idx not in tc_map:
                        tc_map[idx] = {"id": "", "name": "", "args": ""}
                    # 如果当前分片带tool_call id，就记录下来。
                    if tc_delta.id:
                        tc_map[idx]["id"] = tc_delta.id
                    # 如果当前分片带function信息，就继续解析函数名和参数。
                    if tc_delta.function:
                        # function name通常只出现一次，出现时保存。
                        if tc_delta.function.name:
                            tc_map[idx]["name"] = tc_delta.function.name
                        # function arguments是JSON字符串，流式返回时可能分多段，所以持续拼接。
                        if tc_delta.function.arguments:
                            tc_map[idx]["args"] += tc_delta.function.arguments

        # 准备把聚合后的tool_call原始数据转换成ToolCall对象列表。
        parsed: list[ToolCall] = []
        for idx in sorted(tc_map):
            raw = tc_map[idx]
            try:
                args = json.loads(raw["args"])
            except (json.JSONDecodeError, KeyError):
                args = {}
            parsed.append(ToolCall(id=raw["id"], name=raw["name"], arguments=args))

        self.total_prompt_tokens += prompt_tok
        self.total_completion_tokens += completion_tok

        # 返回统一的LLMResponse，供Agent主循环判断是直接回答还是执行工具。
        return LLMResponse(
            content="".join(content_parts),
            tool_calls=parsed,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
        )

    def _call_with_retry(self, params: dict, max_retries: int = 3):
        # 使用LiteLLM发起请求，并对临时错误做指数退避重试。
        """Retry on transient errors with exponential backoff via litellm."""
        # 延迟导入litellm，只有启用LiteLLM后端时才需要安装这个可选依赖。
        import litellm

        # 允许LiteLLM丢弃provider不支持的参数，提升多provider兼容性。
        params["drop_params"] = True
        # 如果配置了api_key，就透传给LiteLLM。
        if self.api_key:
            params["api_key"] = self.api_key
        # 如果配置了base_url，就按LiteLLM命名转换成api_base。
        if self.base_url:
            params["api_base"] = self.base_url

        # 最多尝试max_retries次，默认3次。
        for attempt in range(max_retries):
            # 尝试发起一次LiteLLMcompletion请求。
            try:
                return litellm.completion(**params)
            # LiteLLM可能把不同provider错误包装成普通Exception，所以这里统一从错误文本判断。
            except Exception as e:
                # 转成小写字符串，方便做关键词匹配。
                err = str(e).lower()
                # 判断是否像限流、超时、连接失败或部分临时状态码。
                is_transient = any(
                    kw in err
                    for kw in ["rate_limit", "timeout", "connection", "502", "503", "529"]
                )
                # 判断是否像服务端5xx错误。
                is_server = any(kw in err for kw in ["500", "502", "503", "504"])
                # 如果是可恢复错误且还有重试次数，就指数退避等待后重试。
                if (is_transient or is_server) and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                # 如果不是可恢复错误，或已经用完重试次数，就把异常抛给上层。
                else:
                    raise
