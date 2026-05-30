# Claude Code Provider Proxy

A proxy service that allows Anthropic API requests (especially from [Claude Code](https://github.com/anthropics/claude-code) to be routed through an OpenAI-compatible URL to access alternative models.

![Claude Proxy Logo](docs/cover.png)

## Table of Contents

- [Overview](#overview)
- [Example](#example)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
- [Misc](#misc)
- [Known Working Models](#known-working-models)
- [License](#license)

## Overview

Claude Proxy provides a compatibility layer between [Claude Code](https://github.com/anthropics/claude-code) and alternative models available through [OpenRouter.ai](https://openrouter.ai/) or your chosen base URL. It dynamically reroutes LLM requests from Claude Code to the providers you want to use.

Key features:

- FastAPI web server exposing Anthropic-compatible endpoints
- Format conversion between Anthropic and OpenAI requests/responses
  (see [mapping](docs/mapping.md) for translation details)
- Support for both streaming and non-streaming responses
- Dynamic model selection based on requested Claude model
- Detailed request/response logging
- Token counting

## Example

![Claude Proxy Example](docs/example.png)

## Getting Started

It is recommended to use docker, using uv is recommended for development only.

### Prerequisites

- [Claude Code](https://github.com/anthropics/claude-code) installed (as of July 2025 the proxy is tested on version `1.0.56`, which you can install with `npm install -g @anthropic-ai/claude-code@1.0.56`)
- An [OpenRouter](https://openrouter.ai/) API key
- *Optional if you don't want to use docker*
    - *Python 3.10+*
    - *[uv](https://github.com/astral-sh/uv)*


1. Download the repo: `git clone https://github.com/ujisati/claude-code-provider-proxy/`

2. Either modify the `environment:` in `docker-compose` or create a `.env` file at the root of the repo with for example:

```env
OPENAI_API_KEY=<your openrouter api key>
BIG_MODEL_NAME=google/gemini-2.5-pro-preview
SMALL_MODEL_NAME=google/gemini-2.0-flash-lite-001
LOG_LEVEL=DEBUG
```

For Tuning Engines, use its OpenAI-compatible gateway:

```env
OPENAI_API_KEY=sk-te-your-tuning-engines-key
BASE_URL=https://api.tuningengines.com/v1
BIG_MODEL_NAME=llama-3.3-70b-fp8
SMALL_MODEL_NAME=qwen-2.5-coder-32b
LOG_LEVEL=DEBUG
```

A list of known working models can be found at the bottom of that README.md.

For more configuration options, see the `Settings` class in `src/main.py`.

*Note: you can use environment variables instead of an `.env` file, but note that any value of the `.env` file that is also present in the environment will be overwritten (the environment takes priority). In particular, if you set `OPENAI_API_KEY` in `.env` to the `OPENROUTER_API_KEY`, if you don't `unset OPENAI_API_KEY`, the key received by `src/main.py` will be the one from OpenAI instead of [OpenRouter.ai](https://openrouter.ai/).*

3. Run the proxy:

**Docker: (recommended)**
```bash
# Build and run with docker-compose
docker-compose up --build --detach

# Or build and run manually
docker build -f docker/Dockerfile -t claude-code-proxy .
docker run -p 8080:8080 --env-file .env claude-code-proxy
```

**Local Development:**
```bash
uv run src/main.py
```

4. Run Claude Code

```bash
ANTHROPIC_BASE_URL=http://localhost:8080 claude
```

To make a more permanent alias you can run `echo 'alias claude="ANTHROPIC_BASE_URL=http://localhost:8080 claude"' >> ~/.bashrc` for example.

5. Optional: to further customize `claude`, there are environment variables you can modify, for example:

`CLAUDE_CODE_EXTRA_BODY`
`MAX_THINKING_TOKENS`
`API_TIMEOUT_MS`
`ANTHROPIC_BASE_URL`
`DISABLE_TELEMETRY`
`DISABLE_ERROR_REPORTING`

*Note: These variables modify `claude`, **not** the proxy.*

## Misc

The proxy server exposes the following endpoints:

- `POST /v1/messages`: Create a message (main endpoint)
- `POST /v1/messages/count_tokens`: Count tokens for a request
- `GET /`: Health check endpoint

## Known Working Models

* As `BIG_MODEL_NAME`:
    * `anthropic/claude-sonnet-4`

    * problematic
        * `google/gemini-2.5-pro`: seems to sometimes struggle to respect the edit format expected by `claude`

* As `SMALL_MODEL_NAME`:
    * anthropic/claude-3.5-haiku
    * google/gemini-2.5-flash

## License

[LICENSE](./LICENSE)
