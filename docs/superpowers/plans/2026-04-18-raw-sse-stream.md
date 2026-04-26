# Raw SSE Stream Handler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace OpenAI SDK streaming with raw httpx streaming to extract OpenRouter cost and cache token data from SSE responses.

**Problem:** OpenAI SDK abstracts away raw SSE stream, making it impossible to access OpenRouter-specific usage data (`cost`, `prompt_tokens_details.cached_tokens`, `cache_write_tokens`) that comes in final chunk of SSE stream.

**Architecture:** Use `httpx.AsyncClient` to stream raw SSE lines, parse them manually using regex, then convert to Anthropic format. Keep existing httpx client pool but add new `stream_raw_httpx` function.

**Tech Stack:** Python 3.11, httpx, re, asyncio

---

## Task 1: Create SSE parser utility function

**Files:**
- Modify: `src/main.py` (add `_parse_sse_chunk` helper)

- [ ] **Step 1: Add SSE chunk parser**

After existing imports and before main code:

```python
import re
import httpx

_SSE_LINE_RE = re.compile(r"data: (\{.*\})$", re.MULTILINE)


def _parse_sse_chunk(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single SSE data line into JSON dict. Returns None if not a data line."""
    match = _SSE_LINE_RE.match(line.strip())
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_openrouter_usage(chunk: Dict[str, Any]) -> Tuple[Optional[float], int, int]:
    """Extract cost, cache_create, cache_read from OpenRouter SSE chunk.

    Returns: (cost, cache_create_tokens, cache_read_tokens)
    """
    cost = None
    cache_create = 0
    cache_read = 0

    usage = chunk.get("usage")
    if isinstance(usage, dict):
        # Cost from usage object
        cost_val = usage.get("cost")
        if cost_val is not None:
            cost = float(cost_val)

        # Cache tokens from prompt_tokens_details
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            cache_read = details.get("cached_tokens", 0) or 0
            cache_create = details.get("cache_write_tokens", 0) or 0

    return cost, cache_create, cache_read
```

- [ ] **Step 2: Commit**
```bash
cd claude-code-provider-proxy
git add src/main.py
git commit -m "wip: add SSE parser utility functions"
```

---

## Task 2: Create raw httpx stream handler

**Files:**
- Modify: `src/main.py` (add `handle_anthropic_streaming_from_raw_httpx` function)

- [ ] **Step 1: Create new stream function**

After `handle_anthropic_streaming_response_from_openai_stream`, add:

```python
async def handle_anthropic_streaming_from_raw_httpx(
    httpx_response: httpx.Response,
    original_anthropic_model_name: str,
    estimated_input_tokens: int,
    request_id: str,
    start_time_mono: float,
) -> AsyncGenerator[str, None]:
    """
    Consumes raw httpx SSE stream and yields Anthropic-compatible SSE events.
    Extracts OpenRouter cost and cache tokens from raw chunks.
    """

    anthropic_message_id = f"msg_stream_{request_id}_{uuid.uuid4().hex[:8]}"

    next_anthropic_block_idx = 0
    text_block_anthropic_idx: Optional[int] = None
    openai_tool_idx_to_anthropic_block_idx: Dict[int, int] = {}
    tool_states: Dict[int, Dict[str, Any]] = {}
    sent_tool_block_starts: set[int] = set()

    output_token_count = 0
    final_anthropic_stop_reason: StopReasonType = None
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0

    enc = get_token_encoder(original_anthropic_model_name, request_id)

    openai_to_anthropic_stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "content_filter": "stop_sequence",
        None: None,
    }

    stream_status_code = httpx_response.status_code
    stream_final_message = "Streaming request completed successfully."
    stream_log_event = LogEvent.REQUEST_COMPLETED.value

    try:
        yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': anthropic_message_id, 'type': 'message', 'role': 'assistant', 'model': original_anthropic_model_name, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': estimated_input_tokens, 'output_tokens': 0}}})}\n\n"
        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

        # Read raw SSE lines
        async for line in httpx_response.aiter_lines():
            if not line:
                continue

            # Parse SSE chunk
            chunk_data = _parse_sse_chunk(line)
            if not chunk_data:
                continue

            # Extract OpenRouter usage from raw chunk (even without choices)
            cost, cache_write, cache_read = _extract_openrouter_usage(chunk_data)
            if cost is not None:
                _current_request_cost.set(cost)
            if cache_write:
                cache_creation_input_tokens = cache_write
            if cache_read:
                cache_read_input_tokens = cache_read

            # Process OpenAI-style chunk
            if chunk_data.get("choices"):
                choices = chunk_data["choices"]
                if not choices:
                    continue

                choice = choices[0]
                delta = choice.get("delta", {})
                finish_reason = choice.get("finish_reason")

                # Handle text content
                content = delta.get("content")
                if content:
                    output_token_count += len(enc.encode(content))
                    if text_block_anthropic_idx is None:
                        text_block_anthropic_idx = next_anthropic_block_idx
                        next_anthropic_block_idx += 1
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': text_block_anthropic_idx, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': text_block_anthropic_idx, 'delta': {'type': 'text_delta', 'text': content}})}\n\n"

                # Handle tool calls
                tool_calls = delta.get("tool_calls")
                if tool_calls:
                    for tool_call in tool_calls:
                        tool_idx = tool_call.get("index", 0)
                        if tool_idx not in openai_tool_idx_to_anthropic_block_idx:
                            new_idx = next_anthropic_block_idx
                            next_anthropic_block_idx += 1
                            openai_tool_idx_to_anthropic_block_idx[tool_idx] = new_idx
                            tool_states[tool_idx] = {"name": "", "args": {}}
                            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': new_idx, 'content_block': {'type': 'tool_use', 'id': f'toolu_{tool_idx}', 'name': '', 'input': {}}})}\n\n"

                        anthropic_idx = openai_tool_idx_to_anthropic_block_idx[tool_idx]
                        tool_state = tool_states[tool_idx]
                        name = tool_call.get("function", {}).get("name")
                        args_str = tool_call.get("function", {}).get("arguments")

                        if name:
                            tool_state["name"] = name
                            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_idx, 'content_block': {'type': 'tool_use', 'id': f'toolu_{tool_idx}', 'name': name, 'input': {}}})}\n\n"
                        if args_str:
                            tool_state["args_str"] = tool_state.get("args_str", "") + args_str
                            try:
                                tool_state["args"] = json.loads(tool_state["args_str"])
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': anthropic_idx, 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(tool_state["args"])}})}\n\n"
                            except json.JSONDecodeError:
                                pass

                # Handle finish
                if finish_reason:
                    final_anthropic_stop_reason = openai_to_anthropic_stop_reason_map.get(finish_reason, "end_turn")

            # Check for [DONE]
            if line.strip() == "[DONE]":
                break

        # Yield stop events
        if text_block_anthropic_idx is not None:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': text_block_anthropic_idx})}\n\n"
        for anthropic_idx in openai_tool_idx_to_anthropic_block_idx.values():
            if anthropic_idx not in sent_tool_block_starts:
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_idx, 'content_block': {'type': 'tool_use', 'id': f'toolu_{anthropic_idx}', 'name': tool_state.get('name', ''), 'input': tool_state.get('args', {})}})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': anthropic_idx})}\n\n"

        usage_data = {
            "input_tokens": estimated_input_tokens,
            "output_tokens": output_token_count,
        }
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': final_anthropic_stop_reason}, 'usage': usage_data})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        error_type = "stream_processing_error"
        final_anthropic_stop_reason = "error"
        stream_final_message = f"Error during stream processing: {str(e)}"
        stream_log_event = LogEvent.STREAM_INTERRUPTED.value
        stream_status_code = httpx_response.status_code or 500

        error(
            LogRecord(
                event=LogEvent.STREAM_INTERRUPTED.value,
                message=stream_final_message,
                request_id=request_id,
                data={
                    "error_type": error_type,
                    "traceback": traceback_str,
                },
            ),
            exc=e,
        )
        yield _format_anthropic_error_sse_event(error_type, str(e), None)

    finally:
        duration_ms = (time.monotonic() - start_time_mono) * 1000
        log_data = {
            "status_code": stream_status_code,
            "duration_ms": duration_ms,
            "input_tokens": estimated_input_tokens,
            "output_tokens": output_token_count,
            "stop_reason": final_anthropic_stop_reason,
        }
        cost = _current_request_cost.get()
        if cost is not None:
            log_data["cost"] = cost
        if cache_creation_input_tokens or cache_read_input_tokens:
            log_data["cache_creation_input_tokens"] = cache_creation_input_tokens
            log_data["cache_read_input_tokens"] = cache_read_input_tokens

        if stream_log_event == LogEvent.REQUEST_COMPLETED.value:
            info(
                LogRecord(
                    event=stream_log_event,
                    message=stream_final_message,
                    request_id=request_id,
                    data=log_data,
                )
            )
        else:
            error(
                LogRecord(
                    event=stream_log_event,
                    message=stream_final_message,
                    request_id=request_id,
                    data=log_data,
                )
            )
```

- [ ] **Step 2: Commit**
```bash
cd claude-code-provider-proxy
git add src/main.py
git commit -mfeat: add raw httpx SSE stream handler"
```

---

## Task 3: Integrate raw httpx streaming into main flow

**Files:**
- Modify: `src/main.py` (`create_message_proxy` function)

- [ ] **Step 1: Add raw streaming option**

Find the streaming section (around line 2087), add new path:

```python
            # Try raw httpx streaming for OpenRouter, fallback to OpenAI SDK
            try:
                # Use httpx client directly for raw SSE access
                httpx_client = clients_pool[target_conn.connection_id]
                openai_params_copy = openai_params.copy()
                openai_params_copy["stream"] = True

                # Make request using httpx instead of OpenAI SDK
                httpx_response = await httpx_client.post(
                    target_conn.base_url.rstrip('/') + '/v1/chat/completions',
                    headers={
                        "Authorization": f"Bearer {target_conn.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=openai_params_copy,
                    timeout=180.0,
                )

                if httpx_response.status_code != 200:
                    httpx_response.raise_for_status()

                return StreamingResponse(
                    handle_anthropic_streaming_from_raw_httpx(
                        httpx_response,
                        anthropic_request.model,
                        estimated_input_tokens,
                        request_id,
                        request.state.start_time_monotonic,
                    ),
                    media_type="text/event-stream",
                )

            except Exception as e_httpx:
                # Fallback to OpenAI SDK if raw httpx fails
                debug(
                    LogRecord(
                        LogEvent.REQUEST_FAILURE.value,
                        f"Raw httpx streaming failed, falling back to OpenAI SDK: {str(e_httpx)}",
                        request_id,
                    ),
                )
                openai_stream_response = await target_client.chat.completions.create(**openai_params_copy)
                return StreamingResponse(
                    handle_anthropic_streaming_response_from_openai_stream(
                        openai_stream_response,
                        anthropic_request.model,
                        estimated_input_tokens,
                        request_id,
                        request.state.start_time_monotonic,
                    ),
                    media_type="text/event-stream",
                )
```

- [ ] **Step 2: Update httpx client pool initialization**

Find `clients_pool` initialization (around line 286), ensure httpx client is accessible:

```python
    for conn_id, conn_cfg in proxy_config.connections.items():
        # Keep existing httpx client for raw streaming
        http_client = httpx.AsyncClient(
            event_hooks={'request': [log_request_hook], 'response': [log_response_hook]},
            verify=False,
            timeout=180.0,
            auth=YandexAuth(conn_cfg.api_key),
        )

        # Store httpx client directly (not wrapped in openai client for now)
        # We'll create openai client separately for compatibility
        clients_pool[conn_id] = http_client

        # Create openai client for fallback and non-streaming
        openai_client = openai.AsyncClient(
            api_key=conn_cfg.api_key,
            base_url=conn_cfg.base_url,
            http_client=http_client,
        )
        # Store openai client separately (need new dict)
```

- [ ] **Step 3: Commit**
```bash
cd claude-code-provider-proxy
git add src/main.py
git commit -m"feat: integrate raw httpx streaming with OpenAI SDK fallback"
```

---

## Task 4: Cleanup old OpenAI SDK streaming code

**Files:**
- Modify: `src/main.py` (remove old stream handler, updates)

- [ ] **Step 1: Remove old function after confirming raw stream works**

Once raw stream is verified working, remove `handle_anthropic_streaming_response_from_openai_stream` function and remove fallback to OpenAI SDK in `create_message_proxy`.

- [ ] **Step 2: Commit**
```bash
cd claude-code-provider-proxy
git add src/main.py
git commit -m"refactor: remove old OpenAI SDK streaming code"
```

---

## Task 5: Update formatter for cost/cache display

**Files:**
- Modify: `src/main.py` (`PrettyConsoleFormatter._format_structured` method)

- [ ] **Step 1: Ensure REQUEST_COMPLETED shows cost and cache**

The formatter already has logic for cost and cache display from previous work. Verify it still works with raw stream data.

- [ ] **Step 2: Test end-to-end**

Run proxy and make streaming request. Verify console output shows:
- `$X.XXXX` for cost
- `cache_create=N cache_read=M` for cache tokens (if > 0)

- [ ] **Step 3: Final commit**
```bash
cd claude-code-provider-proxy
git add src/main.py
git commit -m"chore: verify cost/cache display in logs"
```

---

## Task 6: End-to-end verification

- [ ] **Step 1: Test streaming request with OpenRouter**

```bash
cd claude-code-provider-proxy
python -m src.main --verbose
```

Make streaming request to model that routes through OpenRouter. Check logs for:
- Cost information in REQUEST_COMPLETED
- Cache tokens (if relevant)
- No errors in stream processing

- [ ] **Step 2: Test non-streaming request with OpenRouter**

Make non-streaming request. Verify cost/cache are extracted from response body.

- [ ] **Step 3: Test fallback to OpenAI SDK**

Temporarily break raw httpx code and verify fallback to OpenAI SDK works (graceful degradation).

- [ ] **Step 4: Final commit**
```bash
cd claude-code-provider-proxy
git add -A
git commit -m"chore: complete raw SSE stream implementation, verified cost/cache extraction"
```
