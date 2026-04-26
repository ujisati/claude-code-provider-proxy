# Logging Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace raw `print()` calls in HTTPX hooks and request handler with structured LogRecord events, add Rich formatting, concise/verbose modes, and `--log-json` flag.

**Architecture:** Extend the existing LogRecord/PrettyConsoleFormatter system with two new LogEvent types (UPSTREAM_REQUEST, UPSTREAM_RESPONSE). Replace print() calls with _log() calls. Add secret masking, verbose Rich panels, and `--log-json` CLI flag.

**Tech Stack:** Python 3.11, Rich, logging, httpx, pytest

---

### Task 1: Add `mask_secrets` helper + tests

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_log_helpers.py`
- Modify: `src/main.py` (add `mask_secrets` function)

- [ ] **Step 1: Create tests directory and write failing test for `mask_secrets`**

```python
# tests/__init__.py
# empty

# tests/test_log_helpers.py
import sys
sys.path.insert(0, "src")

from main import mask_secrets


def test_mask_secrets_authorization_header():
    headers = {
        "authorization": "OAuth y1__xCapOORpdT-ARiuKyCqndUCxxxxxxxxxxxxxxxxxxxxxxx",
        "content-type": "application/json",
    }
    masked = mask_secrets(headers)
    assert masked["authorization"] == "OAuth y1__...XXX"
    assert masked["content-type"] == "application/json"


def test_mask_secrets_short_value():
    headers = {"x-api-key": "short"}
    masked = mask_secrets(headers)
    assert masked["x-api-key"] == "short...XXX"


def test_mask_secrets_no_secrets():
    headers = {"content-type": "application/json", "accept": "application/json"}
    masked = mask_secrets(headers)
    assert masked == headers


def test_mask_secrets_case_insensitive():
    headers = {"Authorization": "Bearer sk-1234567890abcdef"}
    masked = mask_secrets(headers)
    assert masked["Authorization"] == "Bearer ...XXX"


def test_mask_secrets_x_api_key():
    headers = {"x-api-key": "sk-ant-api03-verylongkey"}
    masked = mask_secrets(headers)
    assert masked["x-api-key"] == "sk-ant-a...XXX"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd claude-code-provider-proxy && python -m pytest tests/test_log_helpers.py -v`
Expected: FAIL — `mask_secrets` not defined

- [ ] **Step 3: Implement `mask_secrets` in `src/main.py`**

Add after `format_log_body` function (around line 119):

```python
_SECRET_HEADER_KEYS = {"authorization", "x-api-key", "api-key", "cookie", "set-cookie", "proxy-authorization"}

def mask_secrets(headers: Dict[str, str]) -> Dict[str, str]:
    """Masks sensitive header values, showing first 8 chars + ...XXX."""
    masked = {}
    for key, value in headers.items():
        if key.lower() in _SECRET_HEADER_KEYS and len(value) > 8:
            masked[key] = value[:8] + "...XXX"
        elif key.lower() in _SECRET_HEADER_KEYS:
            masked[key] = value + "...XXX"
        else:
            masked[key] = value
    return masked
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd claude-code-provider-proxy && python -m pytest tests/test_log_helpers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd claude-code-provider-proxy
git add tests/__init__.py tests/test_log_helpers.py src/main.py
git commit -m "feat: add mask_secrets helper for log header masking"
```

---

### Task 2: Add `extract_last_user_prompt` helper + tests

**Files:**
- Modify: `tests/test_log_helpers.py`
- Modify: `src/main.py` (add `extract_last_user_prompt` function)

- [ ] **Step 1: Write failing test for `extract_last_user_prompt`**

Add to `tests/test_log_helpers.py`:

```python
from main import extract_last_user_prompt


def test_extract_last_user_prompt_from_messages():
    body = {
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "how are you?"},
        ]
    }
    result = extract_last_user_prompt(body)
    assert result == "how are you?"


def test_extract_last_user_prompt_no_user_messages():
    body = {"messages": [{"role": "system", "content": "You are helpful"}]}
    result = extract_last_user_prompt(body)
    assert result == ""


def test_extract_last_user_prompt_truncation():
    body = {
        "messages": [
            {"role": "user", "content": "x" * 200},
        ]
    }
    result = extract_last_user_prompt(body)
    assert len(result) == 80
    assert result.endswith("...")


def test_extract_last_user_prompt_empty_body():
    result = extract_last_user_prompt({})
    assert result == ""


def test_extract_last_user_prompt_content_list():
    body = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "image question"}]},
        ]
    }
    result = extract_last_user_prompt(body)
    assert result == "image question"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd claude-code-provider-proxy && python -m pytest tests/test_log_helpers.py::test_extract_last_user_prompt_from_messages -v`
Expected: FAIL — `extract_last_user_prompt` not defined

- [ ] **Step 3: Implement `extract_last_user_prompt` in `src/main.py`**

Add after `mask_secrets`:

```python
def extract_last_user_prompt(body: Dict[str, Any], max_len: int = 80) -> str:
    """Extracts the last user message content from an OpenAI-style request body."""
    messages = body.get("messages", [])
    last_user_content = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                texts = [
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                last_user_content = " ".join(texts)
            elif isinstance(content, str):
                last_user_content = content
            break
    if len(last_user_content) > max_len:
        return last_user_content[:max_len - 3] + "..."
    return last_user_content
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd claude-code-provider-proxy && python -m pytest tests/test_log_helpers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd claude-code-provider-proxy
git add tests/test_log_helpers.py src/main.py
git commit -m "feat: add extract_last_user_prompt helper for concise log lines"
```

---

### Task 3: Add UPSTREAM_REQUEST/UPSTREAM_RESPONSE LogEvent types + LOG_JSON flag

**Files:**
- Modify: `src/main.py`

- [ ] **Step 1: Add new LogEvent enum members**

In `src/main.py`, add to the `LogEvent` enum (around line 351):

```python
class LogEvent(enum.Enum):
    MODEL_SELECTION = "model_selection"
    REQUEST_START = "request_start"
    REQUEST_COMPLETED = "request_completed"
    REQUEST_FAILURE = "request_failure"
    ANTHROPIC_REQUEST = "anthropic_body"
    OPENAI_REQUEST = "openai_request"
    OPENAI_RESPONSE = "openai_response"
    ANTHROPIC_RESPONSE = "anthropic_response"
    STREAMING_REQUEST = "streaming_request"
    STREAM_INTERRUPTED = "stream_interrupted"
    TOKEN_COUNT = "token_count"
    UPSTREAM_REQUEST = "upstream_request"
    UPSTREAM_RESPONSE = "upstream_response"
    TOKEN_ENCODER_LOAD_FAILED = "token_encoder_load_failed"
    SYSTEM_PROMPT_ADJUSTED = "system_prompt_adjusted"
    TOOL_INPUT_SERIALIZATION_FAILURE = "tool_input_serialization_failure"
    IMAGE_FORMAT_UNSUPPORTED = "image_format_unsupported"
    MESSAGE_FORMAT_NORMALIZED = "message_format_normalized"
    TOOL_RESULT_SERIALIZATION_FAILURE = "tool_result_serialization_failure"
    TOOL_RESULT_PROCESSING = "tool_result_processing"
    TOOL_CHOICE_UNSUPPORTED = "tool_choice_unsupported"
    TOOL_ARGS_TYPE_MISMATCH = "tool_args_type_mismatch"
    TOOL_ARGS_PARSE_FAILURE = "tool_args_parse_failure"
    TOOL_ARGS_UNEXPECTED = "tool_args_unexpected"
    TOOL_ID_PLACEHOLDER = "tool_id_placeholder"
    TOOL_ID_UPDATED = "tool_id_updated"
    PARAMETER_UNSUPPORTED = "parameter_unsupported"
    HEALTH_CHECK = "health_check"
    PROVIDER_ERROR_DETAILS = "provider_error_details"
```

- [ ] **Step 2: Add `LOG_JSON` flag**

After `VERBOSE_LOGGING = "--verbose" in sys.argv` (line 87), add:

```python
LOG_JSON = "--log-json" in sys.argv
```

- [ ] **Step 3: Commit**

```bash
cd claude-code-provider-proxy
git add src/main.py
git commit -m "feat: add UPSTREAM_REQUEST/UPSTREAM_RESPONSE LogEvent types and --log-json flag"
```

---

### Task 4: Replace `log_request_hook` with structured LogRecord

**Files:**
- Modify: `src/main.py` (replace `log_request_hook` function)

- [ ] **Step 1: Replace `log_request_hook` implementation**

Replace the entire `log_request_hook` function (lines 129-131) with:

```python
async def log_request_hook(request: httpx.Request):
    body_str = request.read().decode("utf-8", errors="ignore") if request.stream else ""
    request_id = request.headers.get("x-request-id")

    data: Dict[str, Any] = {
        "method": request.method,
        "url": str(request.url),
    }

    if body_str:
        try:
            parsed_body = json.loads(body_str)
            data["target_model"] = parsed_body.get("model", "?")
            data["stream"] = parsed_body.get("stream", False)
            data["last_user_prompt"] = extract_last_user_prompt(parsed_body)
        except (json.JSONDecodeError, AttributeError):
            data["body_preview"] = body_str[:200] if body_str else ""

    if VERBOSE_LOGGING:
        try:
            parsed_body = json.loads(body_str) if body_str else {}
            data["headers"] = mask_secrets(dict(request.headers))
            data["body"] = truncate_large_structures(parsed_body) if parsed_body else body_str
        except (json.JSONDecodeError, AttributeError):
            data["headers"] = mask_secrets(dict(request.headers))
            data["body"] = body_str

    debug(
        LogRecord(
            event=LogEvent.UPSTREAM_REQUEST.value,
            message=f"PROXY→Provider {request.method} {data.get('target_model', '?')}",
            request_id=request_id,
            data=data,
        )
    )
```

- [ ] **Step 2: Commit**

```bash
cd claude-code-provider-proxy
git add src/main.py
git commit -m "feat: replace log_request_hook print() with structured LogRecord"
```

---

### Task 5: Replace `log_response_hook` with structured LogRecord

**Files:**
- Modify: `src/main.py` (replace `log_response_hook` function)

- [ ] **Step 1: Replace `log_response_hook` implementation**

Replace the entire `log_response_hook` function (lines 133-144) with:

```python
async def log_response_hook(response: httpx.Response):
    request_id = response.request.headers.get("x-request-id") if response.request else None
    content_type = response.headers.get("content-type", "")

    data: Dict[str, Any] = {
        "status_code": response.status_code,
        "url": str(response.url),
    }

    is_sse = "text/event-stream" in content_type
    if is_sse:
        data["body_type"] = "sse_stream"
    else:
        try:
            await response.aread()
            body_text = response.text
            data["body_type"] = "json"
            data["body_preview"] = body_text[:500] if body_text else ""
        except Exception:
            data["body_type"] = "unreadable"

    if VERBOSE_LOGGING:
        data["headers"] = mask_secrets(dict(response.headers))
        if not is_sse:
            try:
                data["body"] = format_log_body(body_text)
            except Exception:
                pass

    debug(
        LogRecord(
            event=LogEvent.UPSTREAM_RESPONSE.value,
            message=f"Provider→PROXY {response.status_code}",
            request_id=request_id,
            data=data,
        )
    )
```

- [ ] **Step 2: Commit**

```bash
cd claude-code-provider-proxy
git add src/main.py
git commit -m "feat: replace log_response_hook print() with structured LogRecord"
```

---

### Task 6: Add UPSTREAM_REQUEST/UPSTREAM_RESPONSE formatting to PrettyConsoleFormatter

**Files:**
- Modify: `src/main.py` (`PrettyConsoleFormatter._format_structured` method)
- Modify: `tests/test_log_helpers.py` (add formatter tests)

- [ ] **Step 1: Write failing tests for new formatter cases**

Add to `tests/test_log_helpers.py`:

```python
import logging
from main import PrettyConsoleFormatter, LogRecord, LogEvent


def _format_record(event: str, message: str, request_id: str = None, data: dict = None) -> str:
    record = logging.LogRecord(
        name="test", level=logging.DEBUG, pathname="", lineno=0,
        msg=message, args=(), exc_info=None,
    )
    record.log_record = LogRecord(event=event, message=message, request_id=request_id, data=data)
    formatter = PrettyConsoleFormatter()
    return formatter.format(record)


def test_format_upstream_request_brief():
    result = _format_record(
        LogEvent.UPSTREAM_REQUEST.value,
        "PROXY→Provider POST minimax-m2.7",
        request_id="a569c2bd12345678",
        data={"target_model": "minimax-m2.7", "stream": True, "last_user_prompt": "привет"},
    )
    assert "PROXY→Provider" in result
    assert "minimax-m2.7" in result
    assert "⚡stream" in result
    assert "привет" in result
    assert "#a569c2bd" in result


def test_format_upstream_response_brief():
    result = _format_record(
        LogEvent.UPSTREAM_RESPONSE.value,
        "Provider→PROXY 200",
        request_id="a569c2bd12345678",
        data={
            "status_code": 200,
            "stop_reason": "end_turn",
            "input_tokens": 27058,
            "output_tokens": 14,
            "duration_ms": 7614,
            "tok_per_sec": 1.8,
        },
    )
    assert "Provider→PROXY" in result
    assert "200" in result
    assert "end_turn" in result
    assert "27058" in result
    assert "14" in result


def test_format_anthropic_request_brief():
    result = _format_record(
        LogEvent.ANTHROPIC_REQUEST.value,
        "Received Anthropic request body",
        request_id="a569c2bd12345678",
        data={"client_model": "claude-haiku-4-5-20251001", "estimated_input_tokens": 27058},
    )
    assert "Claude→Proxy" in result
    assert "claude-haiku" in result
    assert "27058" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd claude-code-provider-proxy && python -m pytest tests/test_log_helpers.py::test_format_upstream_request_brief -v`
Expected: FAIL — new event types not handled in formatter

- [ ] **Step 3: Add UPSTREAM_REQUEST, UPSTREAM_RESPONSE, ANTHROPIC_REQUEST formatting to `_format_structured`**

Add new cases to `PrettyConsoleFormatter._format_structured` (after the existing cases, before the error fallback around line 290):

```python
        if event == LogEvent.UPSTREAM_REQUEST.value:
            target_m = data.get("target_model", "?")
            stream = "⚡stream" if data.get("stream") else "sync"
            prompt = data.get("last_user_prompt", "")
            prompt_display = f' [dim]"{prompt}"[/]' if prompt else ""
            return (
                f"[dim]{ts}[/] [{style}]{badge}[/] {rid}"
                f"[bold cyan]PROXY→Provider[/] {target_m} "
                f"[dim]{stream}[/]{prompt_display}"
            )

        if event == LogEvent.UPSTREAM_RESPONSE.value:
            status = data.get("status_code", "?")
            stop = data.get("stop_reason", "")
            inp = data.get("input_tokens", 0)
            out = data.get("output_tokens", 0)
            dur = data.get("duration_ms", 0)
            tok_s = data.get("tok_per_sec")
            dur_str = f"[{dur_color}]{dur / 1000:.1f}s[/]" if dur else ""
            dur_color = "green" if dur < 5000 else "yellow" if dur < 15000 else "red"
            tok_s_str = f" {tok_s:.1f} tok/s" if tok_s else ""
            return (
                f"[dim]{ts}[/] [{style}]{badge}[/] {rid}"
                f"[bold magenta]Provider→PROXY[/] {status} {stop} "
                f"[dim]in={inp} out={out}[/] {dur_str}{tok_s_str}"
            )

        if event == LogEvent.ANTHROPIC_REQUEST.value:
            client_m = data.get("client_model", "?")
            tokens = data.get("estimated_input_tokens", "?")
            return (
                f"[dim]{ts}[/] [{style}]{badge}[/] {rid}"
                f"[bold cyan]Claude→Proxy[/] {client_m} "
                f"[dim]{tokens} tok[/]"
            )
```

**Note:** The UPSTREAM_RESPONSE case needs `dur_color` calculated BEFORE it's used. Fix the order:

```python
        if event == LogEvent.UPSTREAM_RESPONSE.value:
            status = data.get("status_code", "?")
            stop = data.get("stop_reason", "")
            inp = data.get("input_tokens", 0)
            out = data.get("output_tokens", 0)
            dur = data.get("duration_ms", 0)
            tok_s = data.get("tok_per_sec")
            dur_color = "green" if dur < 5000 else "yellow" if dur < 15000 else "red"
            dur_str = f"[{dur_color}]{dur / 1000:.1f}s[/]" if dur else ""
            tok_s_str = f" {tok_s:.1f} tok/s" if tok_s else ""
            return (
                f"[dim]{ts}[/] [{style}]{badge}[/] {rid}"
                f"[bold magenta]Provider→PROXY[/] {status} {stop} "
                f"[dim]in={inp} out={out}[/] {dur_str}{tok_s_str}"
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd claude-code-provider-proxy && python -m pytest tests/test_log_helpers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd claude-code-provider-proxy
git add tests/test_log_helpers.py src/main.py
git commit -m "feat: add UPSTREAM_REQUEST/UPSTREAM_RESPONSE/ANTHROPIC_REQUEST formatting to PrettyConsoleFormatter"
```

---

### Task 7: Replace Claude→Proxy print() with LogRecord + request_id propagation

**Files:**
- Modify: `src/main.py` (`create_message_proxy` function)

- [ ] **Step 1: Remove `print()` and replace with enriched LogRecord**

In `create_message_proxy` (around line 1676-1686), replace:

```python
        body_str = json.dumps(raw_body, ensure_ascii=False)
        print(f"\n[INCOMING CLAUDE CODE REQUEST] HEADERS: {dict(request.headers)}\nBODY: {format_log_body(body_str)}\n")

        debug(
            LogRecord(
                LogEvent.ANTHROPIC_REQUEST.value,
                "Received Anthropic request body",
                request_id,
                {"body": raw_body},
            )
        )
```

with:

```python
        client_model = raw_body.get("model", "unknown") if isinstance(raw_body, dict) else "unknown"

        anthropic_request_data: Dict[str, Any] = {
            "client_model": client_model,
        }
        if VERBOSE_LOGGING:
            anthropic_request_data["headers"] = mask_secrets(dict(request.headers))
            anthropic_request_data["body"] = raw_body
        else:
            try:
                enc = get_token_encoder(client_model, request_id)
                estimated_tokens_quick = count_tokens_for_anthropic_request(
                    messages=MessagesRequest.model_validate(raw_body, context={"request_id": request_id}).messages,
                    system=MessagesRequest.model_validate(raw_body, context={"request_id": request_id}).system,
                    model_name=client_model,
                    tools=MessagesRequest.model_validate(raw_body, context={"request_id": request_id}).tools,
                    request_id=request_id,
                )
                anthropic_request_data["estimated_input_tokens"] = estimated_tokens_quick
            except Exception:
                anthropic_request_data["estimated_input_tokens"] = "?"

        debug(
            LogRecord(
                LogEvent.ANTHROPIC_REQUEST.value,
                "Received Anthropic request body",
                request_id,
                anthropic_request_data,
            )
        )
```

**Wait** — this double-validates MessagesRequest. That's wasteful. The token count is already calculated later (line 1721). Simpler approach: just log client_model in brief mode, and the token count is already logged separately by `TOKEN_COUNT` event. So for brief mode, we don't need estimated_input_tokens here.

Simpler replacement:

```python
        anthropic_request_data: Dict[str, Any] = {
            "client_model": client_model,
        }
        if VERBOSE_LOGGING:
            body_str = json.dumps(raw_body, ensure_ascii=False)
            anthropic_request_data["headers"] = mask_secrets(dict(request.headers))
            anthropic_request_data["body"] = raw_body

        debug(
            LogRecord(
                LogEvent.ANTHROPIC_REQUEST.value,
                "Received Anthropic request body",
                request_id,
                anthropic_request_data,
            )
        )
```

- [ ] **Step 2: Inject request_id into request scope for httpx hooks**

After `request.state.request_id = request_id` (line 1668), add:

```python
        request.scope.setdefault("headers", [])
        request_id_header = ("x-request-id", request_id.encode("utf-8"))
        request.scope["headers"] = list(request.scope.get("headers", [])) + [request_id_header]
```

**Wait** — this sets it on the INCOMING request scope, but httpx hooks see the OUTGOING httpx request, not the FastAPI request. The request_id needs to be passed to httpx via the outgoing request headers. The cleanest way: add `x-request-id` as a default header to the httpx client, or inject it per-request.

Since the openai client creates the httpx request internally, the best approach is to pass request_id via the httpx client's `default_headers`. But we have a shared client pool.

Better approach: store request_id on `request.state`, and read it from a thread-local or context variable in the hooks. But httpx hooks are async and run in the same event loop, so we can use `contextvars`.

Add near the top of the file:

```python
import contextvars

_current_request_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)
```

In `create_message_proxy`, after generating `request_id`:

```python
        _current_request_id.set(request_id)
```

In `log_request_hook` and `log_response_hook`, replace `request.headers.get("x-request-id")` with:

```python
    request_id = _current_request_id.get()
```

This is the simplest and most reliable approach.

- [ ] **Step 3: Commit**

```bash
cd claude-code-provider-proxy
git add src/main.py
git commit -m "feat: replace Claude→Proxy print() with LogRecord, add request_id context var"
```

---

### Task 8: Replace `log_file_path` with `--log-json` flag

**Files:**
- Modify: `src/main.py` (Settings class, file handler setup)

- [ ] **Step 1: Remove `log_file_path` from Settings, update file handler setup**

In `Settings` class (around line 72), remove `log_file_path`:

```python
class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file="../../.env", extra="ignore")

    config_path: str = "config.yaml"
    referer_url: str = "http://localhost:8080/claude_proxy"

    app_name: str = "AnthropicProxy"
    app_version: str = "0.2.0"
    log_level: str = "DEBUG"
    host: str = "127.0.0.1"
    port: int = 8080
    reload: bool = True
```

Replace the file handler setup (lines 400-411) with:

```python
if LOG_JSON:
    try:
        file_handler = logging.FileHandler("log.jsonl", mode="a")
        file_handler.setFormatter(JSONFormatter())
        _logger.addHandler(file_handler)
    except Exception as e:
        _error_console.print(
            f"Failed to configure JSON log file: {e}"
        )
```

- [ ] **Step 2: Update startup banner**

In the `__main__` block, replace the `Log File` line in the config display (around line 2058):

```python
        ("\n   Log JSON      : ", "default"),
        ("Enabled", "bold green") if LOG_JSON else ("Disabled", "dim"),
```

- [ ] **Step 3: Commit**

```bash
cd claude-code-provider-proxy
git add src/main.py
git commit -m "feat: replace log_file_path setting with --log-json CLI flag"
```

---

### Task 9: Add verbose mode Rich panels for headers/body

**Files:**
- Modify: `src/main.py` (`PrettyConsoleFormatter` class)

- [ ] **Step 1: Add `_format_verbose_panels` helper method to PrettyConsoleFormatter**

Add method to `PrettyConsoleFormatter` class:

```python
    def _format_verbose_panels(self, data: Dict[str, Any]) -> str:
        """Build Rich panel markup for headers and body in verbose mode."""
        parts = []
        headers = data.get("headers")
        if headers:
            header_lines = "\n".join(f"  {k}: {v}" for k, v in headers.items())
            parts.append(
                f"[dim]  ┌─ Headers ─────────────────────────────[/]\n"
                f"{header_lines}\n"
                f"[dim]  └──────────────────────────────────────[/]"
            )
        body = data.get("body")
        if body is not None:
            if isinstance(body, (dict, list)):
                body_str = json.dumps(body, indent=2, ensure_ascii=False)
            else:
                body_str = str(body)
            # Truncate very long bodies in display
            if len(body_str) > 2000:
                body_str = body_str[:1800] + "\n... [truncated] ..."
            body_lines = "\n".join(f"  {line}" for line in body_str.split("\n"))
            parts.append(
                f"[dim]  ┌─ Body ───────────────────────────────[/]\n"
                f"{body_lines}\n"
                f"[dim]  └──────────────────────────────────────[/]"
            )
        return "\n".join(parts)
```

- [ ] **Step 2: Append verbose panels to UPSTREAM_REQUEST, UPSTREAM_RESPONSE, ANTHROPIC_REQUEST formatting**

In `_format_structured`, modify the UPSTREAM_REQUEST case to append verbose panels:

```python
        if event == LogEvent.UPSTREAM_REQUEST.value:
            target_m = data.get("target_model", "?")
            stream = "⚡stream" if data.get("stream") else "sync"
            prompt = data.get("last_user_prompt", "")
            prompt_display = f' [dim]"{prompt}"[/]' if prompt else ""
            base = (
                f"[dim]{ts}[/] [{style}]{badge}[/] {rid}"
                f"[bold cyan]PROXY→Provider[/] {target_m} "
                f"[dim]{stream}[/]{prompt_display}"
            )
            if VERBOSE_LOGGING and (data.get("headers") or data.get("body") is not None):
                return base + "\n" + self._format_verbose_panels(data)
            return base
```

Similarly for UPSTREAM_RESPONSE:

```python
        if event == LogEvent.UPSTREAM_RESPONSE.value:
            status = data.get("status_code", "?")
            stop = data.get("stop_reason", "")
            inp = data.get("input_tokens", 0)
            out = data.get("output_tokens", 0)
            dur = data.get("duration_ms", 0)
            tok_s = data.get("tok_per_sec")
            dur_color = "green" if dur < 5000 else "yellow" if dur < 15000 else "red"
            dur_str = f"[{dur_color}]{dur / 1000:.1f}s[/]" if dur else ""
            tok_s_str = f" {tok_s:.1f} tok/s" if tok_s else ""
            base = (
                f"[dim]{ts}[/] [{style}]{badge}[/] {rid}"
                f"[bold magenta]Provider→PROXY[/] {status} {stop} "
                f"[dim]in={inp} out={out}[/] {dur_str}{tok_s_str}"
            )
            if VERBOSE_LOGGING and (data.get("headers") or data.get("body") is not None):
                return base + "\n" + self._format_verbose_panels(data)
            return base
```

And for ANTHROPIC_REQUEST:

```python
        if event == LogEvent.ANTHROPIC_REQUEST.value:
            client_m = data.get("client_model", "?")
            tokens = data.get("estimated_input_tokens", "")
            tokens_display = f" [dim]{tokens} tok[/]" if tokens else ""
            base = (
                f"[dim]{ts}[/] [{style}]{badge}[/] {rid}"
                f"[bold cyan]Claude→Proxy[/] {client_m}{tokens_display}"
            )
            if VERBOSE_LOGGING and (data.get("headers") or data.get("body") is not None):
                return base + "\n" + self._format_verbose_panels(data)
            return base
```

- [ ] **Step 3: Commit**

```bash
cd claude-code-provider-proxy
git add src/main.py
git commit -m "feat: add verbose mode Rich panels for headers/body in log output"
```

---

### Task 10: Populate UPSTREAM_RESPONSE data with tokens/duration/tok_per_sec

**Files:**
- Modify: `src/main.py` (stream handler and non-stream handler)

- [ ] **Step 1: Pass token and duration data to UPSTREAM_RESPONSE log**

The UPSTREAM_RESPONSE hook fires when the HTTP response arrives, but token counts and duration are only known later (after stream processing or in REQUEST_COMPLETED). The hook only has HTTP-level data (status code, headers).

For **non-streaming** responses: the hook sees the full response body, so we can extract usage from it. Update `log_response_hook` for non-SSE responses:

```python
    if not is_sse:
        try:
            await response.aread()
            body_text = response.text
            data["body_type"] = "json"
            try:
                parsed = json.loads(body_text)
                usage = parsed.get("usage", {})
                data["input_tokens"] = usage.get("prompt_tokens", 0)
                data["output_tokens"] = usage.get("completion_tokens", 0)
            except (json.JSONDecodeError, AttributeError):
                pass
            data["body_preview"] = body_text[:500] if body_text else ""
        except Exception:
            data["body_type"] = "unreadable"
```

For **SSE** responses: token data is not available at HTTP level. It will be logged separately by REQUEST_COMPLETED. The UPSTREAM_RESPONSE brief line will show status + `<SSE stream>`:

```python
    if event == LogEvent.UPSTREAM_RESPONSE.value:
        ...
        body_type = data.get("body_type", "")
        if body_type == "sse_stream":
            return (
                f"[dim]{ts}[/] [{style}]{badge}[/] {rid}"
                f"[bold magenta]Provider→PROXY[/] {status} "
                f"[dim]<SSE stream>[/]"
            )
        # non-SSE response with full data
        ...
```

- [ ] **Step 2: Commit**

```bash
cd claude-code-provider-proxy
git add src/main.py
git commit -m "feat: populate UPSTREAM_RESPONSE with token data for non-SSE responses"
```

---

### Task 11: End-to-end verification

**Files:**
- No changes

- [ ] **Step 1: Start proxy without flags, send a test request, verify brief log format**

Run: `cd claude-code-provider-proxy && python -m src.main`
Then: `curl -X POST http://127.0.0.1:8080/v1/messages -H "x-api-key: my-secret-proxy-key" -H "content-type: application/json" -d '{"model":"claude-haiku-4-5-20251001","messages":[{"role":"user","content":"hi"}],"max_tokens":10,"stream":true}'`

Expected console output includes:
- `Claude→Proxy claude-haiku-4-5-20251001` line
- `PROXY→Provider minimax-m2.7 ⚡stream "hi"` line
- `Provider→PROXY 200 <SSE stream>` line
- NO raw `[HTTPX OUTGOING]` / `[INCOMING CLAUDE CODE REQUEST]` print output
- NO `log.jsonl` file created

- [ ] **Step 2: Start proxy with --verbose, verify Rich panels appear**

Run: `cd claude-code-provider-proxy && python -m src.main --verbose`
Send same curl request. Expected: brief lines + Rich panels with Headers/Body sections below them.

- [ ] **Step 3: Start proxy with --log-json, verify JSONL file is created**

Run: `cd claude-code-provider-proxy && python -m src.main --log-json`
Send curl request. Expected: `log.jsonl` file created with JSON entries for all events.

- [ ] **Step 4: Start proxy with --verbose --log-json, verify both work together**

Run: `cd claude-code-provider-proxy && python -m src.main --verbose --log-json`
Expected: Rich panels in console + full data in JSONL.

- [ ] **Step 5: Final commit**

```bash
cd claude-code-provider-proxy
git add -A
git commit -m "chore: verify logging redesign end-to-end"
```
