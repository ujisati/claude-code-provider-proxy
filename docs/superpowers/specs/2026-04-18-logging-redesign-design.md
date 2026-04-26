# Logging Redesign Design

## Problem

Current logging has three issues:
1. HTTPX hooks (`log_request_hook`, `log_response_hook`) use raw `print()` — no Rich formatting, no request_id, no colors
2. Terminology is confusing: OUTGOING/INCOMING doesn't clearly express the proxy's perspective
3. Duplicate logging: `print()` on line 1677 duplicates the structured `LogRecord` for ANTHROPIC_REQUEST

## Decisions

### Terminology
- `PROXY → Provider` — outgoing request from proxy to upstream provider
- `Provider → PROXY` — incoming response from upstream provider to proxy
- `Claude → Proxy` — incoming request from Claude Code client

### Verbosity levels
- **Default (non-verbose):** concise one-line logs
- **`--verbose`:** full headers + bodies in Rich-formatted panels

### New CLI flag: `--log-json`
- Creates `log.jsonl` file handler with full data (headers, bodies, etc.)
- Without flag: no file handler created
- Replaces `log_file_path` setting in `Settings`

## Architecture

### New LogEvent types
- `UPSTREAM_REQUEST` — request from proxy to upstream provider
- `UPSTREAM_RESPONSE` — response from upstream provider to proxy

### Request ID propagation
Middleware already sets `X-Request-ID` on responses. Need to also inject it into httpx requests so hooks can read it:
1. In `create_message_proxy`, after generating `request_id`, add it to `request.scope['headers']`
2. In httpx hooks, read `request.headers.get("x-request-id")`

### Remove
- All `print()` calls in `log_request_hook`, `log_response_hook`, and `create_message_proxy` (line 1677)
- `log_file_path` from `Settings` class

### Add
- `LOG_JSON = "--log-json" in sys.argv` (like existing `VERBOSE_LOGGING`)
- `UPSTREAM_REQUEST` and `UPSTREAM_RESPONSE` to `LogEvent` enum
- Rich formatting in `PrettyConsoleFormatter._format_structured()` for new events
- `format_rich_headers()` and `format_rich_body()` helpers for verbose mode panels

## Format specifications

### Brief mode (default)

**Claude → Proxy (ANTHROPIC_REQUEST):**
```
19:32:35.282 DBG #a569c2bd Claude→Proxy claude-haiku-4-5-20251001 27058 tok
```

**PROXY → Provider (UPSTREAM_REQUEST):**
```
19:32:35.301 INF #a569c2bd PROXY→Provider minimax-m2.7 ⚡stream "привет"
```
- target_model, stream/sync, last user prompt (truncated ~80 chars)

**Provider → PROXY (UPSTREAM_RESPONSE):**
```
19:32:42.833 INF #a569c2bd Provider→PROXY 200 end_turn in=27058 out=14 7.6s 1.8 tok/s
```
- status, stop_reason, input/output tokens, duration, tok/s speed
- Cost shown only if available from headers

### Verbose mode (--verbose)

All brief-mode lines PLUS Rich panels below them:

**PROXY → Provider:**
```
19:32:35.301 DBG #a569c2bd PROXY→Provider minimax-m2.7 ⚡stream "привет"
  ┌─ Headers ─────────────────────────────
  │ authorization: OAuth y1__...XXX
  │ content-type: application/json
  │ ...
  └─ Body ───────────────────────────────
  {
    "messages": [...],
    "model": "minimax/minimax-m2.7",
    ...
  }
```

**Provider → PROXY:**
- Same panel format for headers and body (or `<SSE stream>`)

**Claude → Proxy:**
- Same panel format for incoming request headers and body

### Secret masking
- Authorization/API key values: show first 8 chars + `...XXX`
- Applied in both console and JSONL output

### JSONL file (--log-json)
- Always writes full data regardless of `--verbose`
- Path: `log.jsonl` in working directory
- Uses existing `JSONFormatter`
- All LogRecord events written, including UPSTREAM_REQUEST/UPSTREAM_RESPONSE with full headers and bodies

## Implementation changes summary

### `log_request_hook` (currently line 129)
- Replace `print()` with `_log(DEBUG, LogRecord(event=UPSTREAM_REQUEST, ...))`
- Extract target_model from body, last user prompt from messages
- Pass request_id from `request.headers.get("x-request-id")`

### `log_response_hook` (currently line 133)
- Replace `print()` with `_log(DEBUG, LogRecord(event=UPSTREAM_RESPONSE, ...))`
- Calculate duration, tok/s from available data
- Pass request_id from `request.headers.get("x-request-id")`

### `create_message_proxy` (line 1677)
- Remove `print()` call
- Inject request_id into request scope for httpx hooks

### `PrettyConsoleFormatter._format_structured`
- Add cases for `UPSTREAM_REQUEST` and `UPSTREAM_RESPONSE`
- In verbose mode, append Rich panels for headers/body

### `Settings` class
- Remove `log_file_path`
- Add `LOG_JSON` module-level flag

### File handler setup (around line 400)
- Guard with `LOG_JSON` flag instead of `settings.log_file_path`
