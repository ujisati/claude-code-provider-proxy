# Model Connections Implementation Plan

> **For Antigravity:** REQUIRED WORKFLOW: Use `.agent/workflows/execute-plan.md` to execute this plan in single-flow mode.

**Goal:** Modify the proxy server to support a generic pool of OpenAI-compatible connections configured via a YAML file, mapping Anthropic models into big, medium, and small categories pointing to distinct connections.

**Architecture:** We will replace the flat `BIG_MODEL_NAME`/`SMALL_MODEL_NAME` env vars with a structured `config.yaml` file loaded via `PyYAML` and validated by `Pydantic`. The proxy will manage a pool of `AsyncClient`s for each configured connection and dynamically route requests depending on whether user requested Opus, Sonnet, or Haiku.

**Tech Stack:** FastAPI, Pydantic, PyYAML, OpenAI Python Client

---

### Task 1: Add Dependency and update .gitignore

**Files:**
- Modify: `pyproject.toml:7-16`
- Modify: `claude-code-provider-proxy/.gitignore`
- Create: `claude-code-provider-proxy/config.yaml.example`

**Step 1: Write the config.yaml.example**

```yaml
proxy_api_key: "my-secret-proxy-key"

mappings:
  big_model: openrouter_gemini_pro
  medium_model: openrouter_gemini_flash
  small_model: local_llama

connections:
  openrouter_gemini_pro:
    base_url: "https://openrouter.ai/api/v1"
    api_key: "sk-or-v1-YOUR_KEY"
    target_model: "google/gemini-2.5-pro"

  openrouter_gemini_flash:
    base_url: "https://openrouter.ai/api/v1"
    api_key: "sk-or-v1-YOUR_KEY"
    target_model: "google/gemini-2.0-flash-lite-001"

  local_llama:
    base_url: "http://localhost:11434/v1"
    api_key: "ollama"
    target_model: "llama3"
```

**Step 2: Add PyYAML dependency**

Run: `uv add pyyaml`
Expected: `pyyaml` added to `pyproject.toml` dependencies

**Step 3: Update .gitignore**

Run: `echo "\nconfig.yaml" >> .gitignore`
Expected: `config.yaml` ignored in git.

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock .gitignore config.yaml.example
git commit -m "chore: add pyyaml dependency and config example"
```

### Task 2: Refactor Settings and load Config Models in main.py

**Files:**
- Modify: `claude-code-provider-proxy/src/main.py:38-84` (Settings initialization)

**Step 1: Write Pydantic Models for Configuration**

```python
import yaml

class ConnectionConfig(BaseModel):
    base_url: str
    api_key: str
    target_model: str

class MappingsConfig(BaseModel):
    big_model: str
    medium_model: str
    small_model: str

class ProxyConfig(BaseModel):
    proxy_api_key: Optional[str] = None
    mappings: MappingsConfig
    connections: Dict[str, ConnectionConfig]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../../.env", extra="ignore")

    config_path: str = "config.yaml"
    referer_url: str = "http://localhost:8080/claude_proxy"

    app_name: str = "AnthropicProxy"
    app_version: str = "0.2.0"
    log_level: str = "INFO"
    log_file_path: Optional[str] = "log.jsonl"
    host: str = "127.0.0.1"
    port: int = 8080
    reload: bool = True

settings = Settings()

proxy_config: ProxyConfig
clients_pool: Dict[str, openai.AsyncClient] = {}

def load_proxy_config() -> None:
    global proxy_config, clients_pool
    try:
        with open(settings.config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        proxy_config = ProxyConfig(**data)
    except Exception as e:
        _error_console.print(f"[bold red]Configuration Error:[/bold red] Failed to load {settings.config_path}: {e}")
        sys.exit(1)

    for conn_id, conn_cfg in proxy_config.connections.items():
        clients_pool[conn_id] = openai.AsyncClient(
            api_key=conn_cfg.api_key,
            base_url=conn_cfg.base_url,
            default_headers={
                "HTTP-Referer": settings.referer_url,
                "X-Title": settings.app_name,
            },
            timeout=180.0,
        )

load_proxy_config()
```

**Step 2: Replace old `openai_client` initialization in main.py**

Remove lines 467-485 in `src/main.py` where global `openai_client` was previously instantiated.

**Step 3: Run the proxy briefly to ensure no syntax errors**

Run: `uv run src/main.py` (Ctrl+C after confirming it boots or fails cleanly on file absence). 

**Step 4: Commit**

```bash
git add src/main.py
git commit -m "feat: setup dynamic yaml configuration and connection pool"
```

### Task 3: Update target routing and client usage in Request Handlers

**Files:**
- Modify: `claude-code-provider-proxy/src/main.py:1390-1426` (`select_target_model`)
- Modify: `claude-code-provider-proxy/src/main.py:1500-1800` (Anywhere `openai_client` and `select_target_model` are called, routing logic)

**Step 1: Rewrite select_target_model to return `(ConnectionConfig, openai.AsyncClient)`**

```python
def select_connection(client_model_name: str, request_id: str) -> Tuple[ConnectionConfig, openai.AsyncClient]:
    """Selects the target connection config and client based on the client's request."""
    client_model_lower = client_model_name.lower()
    
    if "opus" in client_model_lower:
        target_conn_id = proxy_config.mappings.big_model
    elif "sonnet" in client_model_lower:
        target_conn_id = proxy_config.mappings.medium_model
    elif "haiku" in client_model_lower:
        target_conn_id = proxy_config.mappings.small_model
    else:
        target_conn_id = proxy_config.mappings.small_model
        warning(
            LogRecord(
                event=LogEvent.MODEL_SELECTION.value,
                message=f"Unknown client model '{client_model_name}', defaulting to SMALL mapping '{target_conn_id}'.",
                request_id=request_id,
            )
        )
    
    if target_conn_id not in proxy_config.connections:
        error_msg = f"Target connection '{target_conn_id}' is not defined in config.yaml"
        critical(LogRecord(event=LogEvent.MODEL_SELECTION.value, message=error_msg))
        raise Exception(error_msg)
        
    return proxy_config.connections[target_conn_id], clients_pool[target_conn_id]
```

**Step 2: Update endpoints to utilize the new pool and add Authentication**

In `/v1/messages` and `/v1/messages/count_tokens`:
1. Add a FastAPI dependency or simple header check to read `x-api-key`.
2. Check if `proxy_config.proxy_api_key` matches the incoming header. If not, raise `_build_anthropic_error_response(AnthropicErrorType.AUTHENTICATION, "Invalid API key", 401)`.
3. Change usages of the global client:
```python
# From:
# target_model = select_target_model(req.model, request_id)
# ... openai_client.chat.completions.create(...)

# To:
# target_conn, target_client = select_connection(req.model, request_id)
# target_model = target_conn.target_model
# ... target_client.chat.completions.create(...)
```

**Step 3: Write tests/manual verification**

Run the proxy with `uv run src/main.py` using `.env` pointing to example config.
Execute `ANTHROPIC_BASE_URL=http://localhost:8080 claude -p "sonnet"` and verify logs.

**Step 4: Commit**

```bash
git add src/main.py
git commit -m "feat: route requests to dynamically configured connections"
```
