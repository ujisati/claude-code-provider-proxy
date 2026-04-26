# Claude Code Provider Proxy — Agent Context

## Назначение

Прокси-сервер, который позволяет использовать **Claude Code** (официальный CLI/IDE-агент от Anthropic) с **любыми LLM-провайдерами** через OpenAI-совместимый API. Claude Code отправляет запросы в формате Anthropic API — прокси перехватывает их, конвертирует в формат OpenAI Chat Completions, маршрутизирует на выбранную модель и конвертирует ответ обратно в формат Anthropic.

## Архитектура

```
Claude Code  ──►  Proxy (FastAPI, :8080)  ──►  OpenAI-compatible API (OpenRouter / Yandex internal / etc.)
  Anthropic API         ↕ конвертация             OpenAI Chat Completions API
                   config.yaml (маппинг моделей)
```

**Монолитный single-file сервер**: вся логика в одном файле `src/main.py` (~2000 строк).

## Структура проекта

```
Claude_Code_Proxy/
├── .env                          # ANTHROPIC_BASE_URL + ANTHROPIC_API_KEY (автогенерируется при запуске)
├── .agent/skills/                # ⭐ Локальные скиллы агентов (архитектурные паттерны)
├── config.yaml                   # ⭐ Конфигурация: маппинги моделей и подключения
├── pyproject.toml                # Зависимости (uv/pip)
├── log.jsonl                     # Файловый лог (JSON, append-only)
├── Claude_Code_Proxy.code-workspace # VSCode workspace конфигурация
├── src/
│   └── main.py                   # ⭐ Весь серверный код
├── sandbox/                      # Вспомогательные скрипты и кэши
│   ├── update_models.py          # Скрипт обновления списка моделей
│   ├── test_openrouter.py        # Тестовый скрипт
│   ├── internal_models.json      # Кэш внутренних моделей Yandex
│   └── models.json               # Кэш отфильтрованных моделей
├── docs/                         # Документация по проекту
└── docker/                       # Docker-конфигурация
```

## Ключевые файлы

### `config.yaml` — Конфигурация маппинга моделей

```yaml
proxy_api_key: "my-secret-proxy-key"    # Ключ для аутентификации запросов от Claude Code

mappings:
  big_model: opus_4_7              # claude-opus-* → connection_id
  medium_model: qwen-3.6-plus     # claude-sonnet-* → connection_id  
  small_model: minimax_m2.7       # claude-haiku-* → connection_id

connections:
  opus_4_7:                         # connection_id
    base_url: "https://..."         # OpenAI-compatible endpoint
    api_key: "..."                  # API ключ провайдера
    target_model: "anthropic/claude-opus-4.7"  # Модель на стороне провайдера
    provider: ["Anthropic"]         # (опц.) OpenRouter provider routing
```

**Логика маппинга** (`select_connection`): имя модели от Claude Code проверяется на подстроки `opus` / `sonnet` / `haiku` → выбирается соответствующий `big_model` / `medium_model` / `small_model` → находится connection по ID.

### `src/main.py` — Серверный код

Файл содержит следующие логические блоки (сверху вниз):

| Строки (примерно) | Блок | Описание |
|---|---|---|
| 1–43 | **Импорты и инициализация** | Зависимости, `setproctitle("claude-code-proxy")` |
| 45–106 | **Конфигурация** | Pydantic-модели `ConnectionConfig`, `MappingsConfig`, `ProxyConfig`, `Settings`; загрузка `config.yaml`; создание пула `openai.AsyncClient` |
| 109–362 | **Логирование** | `JSONFormatter` (файл), `PrettyConsoleFormatter` + `_RichHandler` (консоль, цветной вывод через Rich), `LogEvent` enum, хелперы `debug/info/warning/error/critical` |
| 387–542 | **Pydantic-модели данных** | Anthropic request/response модели: `MessagesRequest`, `MessagesResponse`, `ContentBlock*`, `Tool`, `AnthropicError*`, `Usage` |
| 545–620 | **Утилиты** | `extract_provider_error_details`, `get_token_encoder` (tiktoken), `count_tokens_for_anthropic_request` |
| 622–1076 | **Конвертация Anthropic → OpenAI** | `convert_anthropic_to_openai_messages`, `convert_anthropic_tools_to_openai`, `convert_anthropic_tool_choice_to_openai` |
| 1076–1470 | **Конвертация OpenAI → Anthropic** | `convert_openai_to_anthropic_response`, `handle_anthropic_streaming_response_from_openai_stream` (SSE стриминг) |
| 1478–1575 | **Бизнес-логика** | `select_connection`, `_build_anthropic_error_response`, `_log_and_return_error_response`, `_get_anthropic_error_details_from_exc` |
| 1577–1860 | **FastAPI эндпоинты** | `GET /v1/models`, `POST /v1/messages` (основной), `POST /v1/messages/count_tokens`, `GET /` (health) |
| 1880–1920 | **Exception handlers** | Глобальные обработчики `openai.APIError`, `ValidationError`, `JSONDecodeError`, `Exception` |
| 1923–1938 | **Middleware** | `logging_middleware` — добавляет `X-Request-ID`, `X-Response-Time-ms` |
| 1941–2012 | **Запуск** | `update_env_file()`, ASCII-баннер, `uvicorn.run()` |

### `update_models.py` — Утилита обновления моделей

Скрипт для получения списка доступных моделей с API Yandex (`api.eliza.yandex.net/models`). Фильтрует мусор (test, embeddings, image, audio и т.д.), оставляет только топовые LLM, сохраняет в `models.json` и `internal_models.json`.

## Запуск

```bash
uv run python src/main.py
```

Процесс регистрируется как `claude-code-proxy` в системе. Слушает на `http://127.0.0.1:8080`. При старте автоматически обновляет `.env` в корне проекта с `ANTHROPIC_BASE_URL` и `ANTHROPIC_API_KEY`.

## API эндпоинты

| Метод | Путь | Описание |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/v1/models` | Фейковый список моделей для Claude Code |
| `POST` | `/v1/messages` | ⭐ Основной: принимает Anthropic Messages API, проксирует через OpenAI API |
| `POST` | `/v1/messages/count_tokens` | Подсчёт токенов (tiktoken) |

## Логирование

**Два канала:**
- **Консоль** (`stdout`): цветной human-readable формат через Rich (`PrettyConsoleFormatter`)
- **Файл** (`log.jsonl`): полный JSON для анализа (`JSONFormatter`)

**Формат консольного лога:**
```
18:24:01.225 INF #90d99a3f claude-sonnet-4-6 → qwen/qwen3.6-plus (⚡stream, ~7 tok)
18:24:02.799 ERR #90d99a3f FAIL [claude-haiku-4] Request failed: Invalid API key (authentication_error, 1ms)
18:24:05.100 INF #90d99a3f 4523ms in=1200 out=350 stop=end_turn $0.0014 prov=OpenAI
```

## Стек технологий

- **Python 3.10+**, **uv** (пакетный менеджер)
- **FastAPI** + **Uvicorn** (ASGI-сервер)
- **OpenAI Python SDK** (клиент к upstream)
- **Pydantic v2** (валидация)
- **Rich** (консольный вывод)
- **tiktoken** (подсчёт токенов)
- **PyYAML** (конфиг)

## Ключевые паттерны для агентов

1. **Всё в одном файле**: не ищи модули — весь код в `src/main.py`.
2. **Конфигурация через `config.yaml`**: добавление нового провайдера = новая запись в `connections` + обновление `mappings`.
3. **Конвертация форматов**: Anthropic ↔ OpenAI — это ядро прокси. Самые сложные части: стриминг (SSE) и tool use.
4. **Аутентификация**: проверка `x-api-key` заголовка против `proxy_api_key` из конфига. Проверка происходит ПОСЛЕ парсинга тела запроса (чтобы в логах видеть модель даже при 401).
5. **OpenRouter provider routing**: передаётся через `extra_body={"provider": {"order": [...]}}` в OpenAI SDK.
6. **Логирование**: структурированное через dataclass `LogRecord` + enum `LogEvent`. Для добавления нового типа лога — добавь значение в `LogEvent` и обработку в `PrettyConsoleFormatter._format_structured`.
7. **OpenRouter Metadata & SSE**: OpenRouter добавляет нестандартные поля (`provider`, `usage.cost`, `cached_tokens`) в SSE-чанки. Они извлекаются как напрямую из `httpx` потока, так и через `chunk.model_extra.get(...)` при работе с OpenAI SDK.
8. **Управление состоянием (ContextVars)**: Для проброса данных (стоимость, провайдер) через асинхронные хуки `httpx` используются `contextvars.ContextVar` (`_current_request_cost`, `_current_request_provider`).
9. **Скиллы агентов**: Обязательно проверяй папку `.agent/skills/` для получения подробной архитектурной документации перед модификацией логики прокси.
