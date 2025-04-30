"""
FastAPI application definition, endpoints, and request handling logic.
Orchestrates calls to other components (config, models, conversion, etc.).
"""

import json
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union, cast

import fastapi
import openai
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from pydantic import ValidationError

from . import conversion, logger, models, token_counter
from .models import extract_provider_error_details, STATUS_CODE_ERROR_MAP
from .config import settings
from .logger import LogEvent, LogRecord
from .openrouter_client import client
from .provider_mods import apply_provider_modifications

app = fastapi.FastAPI(
    title=settings.app_name,
    description="Routes Anthropic API requests to OpenRouter, selecting models dynamically.",
    version=settings.app_version,
    docs_url=None,
    redoc_url=None,
)


def select_target_model(client_model_name: str, request_id: str) -> str:
    """Selects the target OpenRouter model based on the client's request."""
    client_model_lower = client_model_name.lower()

    if "opus" in client_model_lower or "sonnet" in client_model_lower:
        target_model = settings.big_model_name
        logger.debug(
            LogRecord(
                event=LogEvent.MODEL_SELECTION.value,
                message="Client model mapped to BIG model",
                request_id=request_id,
                data={"client_model": client_model_name, "target_model": target_model},
            )
        )
    elif "haiku" in client_model_lower:
        target_model = settings.small_model_name
        logger.debug(
            LogRecord(
                event=LogEvent.MODEL_SELECTION.value,
                message="Client model mapped to SMALL model",
                request_id=request_id,
                data={"client_model": client_model_name, "target_model": target_model},
            )
        )
    else:
        target_model = settings.small_model_name
        logger.warning(
            LogRecord(
                event=LogEvent.MODEL_SELECTION.value,
                message="Unknown client model, defaulting to SMALL model",
                request_id=request_id,
                data={"client_model": client_model_name, "target_model": target_model},
            )
        )
    return target_model


@app.post("/v1/messages", response_model=None, tags=["API"], status_code=200)
async def create_message(request: Request) -> Union[JSONResponse, StreamingResponse]:
    """
    Main endpoint for Anthropic message completions, proxied to OpenRouter.
    Handles conversions, streaming, dynamic model selection, and provider mods.
    Logs requests and errors with traceable IDs.
    """
    request_id = str(uuid.uuid4()).split("-")[0]
    request.state.request_id = request_id
    start_time = time.time()
    is_stream = False

    body = await request.json()

    anthropic_request = models.MessagesRequest.model_validate(body)
    is_stream = anthropic_request.stream or False

    target_model = select_target_model(anthropic_request.model, request_id)
    target_provider = (
        target_model.split("/")[0].lower() if "/" in target_model else "unknown"
    )
    logger.debug(
        LogRecord(
            event=LogEvent.PROVIDER_IDENTIFICATION.value,
            message="Target provider identified",
            request_id=request_id,
            data={"provider": target_provider},
        )
    )

    logger.debug(
        LogRecord(
            event=LogEvent.ANTHROPIC_REQUEST.value,
            message="Anthropic request body",
            request_id=request_id,
            data={"body": body},
        )
    )

    openai_messages = conversion.convert_anthropic_to_openai_messages(
        anthropic_request.messages, anthropic_request.system, request_id=request_id
    )
    openai_tools = conversion.convert_anthropic_tools_to_openai(anthropic_request.tools)
    openai_tool_choice = conversion.convert_anthropic_tool_choice_to_openai(
        anthropic_request.tool_choice, request_id=request_id
    )

    estimated_input_tokens = 0

    # Calculate input tokens before branching
    estimated_input_tokens = token_counter.count_tokens_for_request(
        messages=anthropic_request.messages,
        system=anthropic_request.system,
        model_name=anthropic_request.model,
        tools=anthropic_request.tools,
        request_id=request_id,
    )

    logger.info(
        LogRecord(
            event=LogEvent.REQUEST_START.value,
            message="Request started",
            request_id=request_id,
            data={
                "client_model": anthropic_request.model,
                "target_model": target_model,
                "stream": is_stream,
                "estimated_input_tokens": estimated_input_tokens,
            },
        )
    )

    messages = cast(List[ChatCompletionMessageParam], openai_messages)
    tools = cast(Optional[List[ChatCompletionToolParam]], openai_tools)

    openai_params: Dict[str, Any] = {
        "model": target_model,
        "messages": messages,
        "max_tokens": anthropic_request.max_tokens,
        "stream": is_stream,
    }

    if anthropic_request.temperature is not None:
        openai_params["temperature"] = anthropic_request.temperature
    if anthropic_request.top_p is not None:
        openai_params["top_p"] = anthropic_request.top_p
    if anthropic_request.stop_sequences:
        openai_params["stop"] = anthropic_request.stop_sequences
    if tools:
        openai_params["tools"] = tools
    if openai_tool_choice is not None:
        openai_params["tool_choice"] = openai_tool_choice
    if anthropic_request.metadata and anthropic_request.metadata.get("user_id"):
        openai_params["user"] = str(anthropic_request.metadata.get("user_id"))

    openai_params = apply_provider_modifications(
        openai_params, target_provider, request_id=request_id
    )

    logger.debug(
        LogRecord(
            event=LogEvent.OPENAI_REQUEST.value,
            message="OpenAI request parameters",
            request_id=request_id,
            data={"params": openai_params},
        )
    )

    if is_stream:
        logger.debug(
            LogRecord(
                event=LogEvent.STREAMING_REQUEST.value,
                message="Initiating streaming request to OpenRouter",
                request_id=request_id,
            )
        )
        response_generator = await client.chat.completions.create(**openai_params)

        return StreamingResponse(
            conversion.handle_streaming_response(
                response_generator,
                anthropic_request.model,
                estimated_input_tokens,
                request_id,
                start_time,
            ),
            media_type="text/event-stream",
        )
    else:
        logger.debug(
            LogRecord(
                event=LogEvent.OPENAI_REQUEST.value,
                message="Sending non-streaming request to OpenRouter",
                request_id=request_id,
            )
        )
        openai_response = await client.chat.completions.create(**openai_params)

        response_dict = openai_response.model_dump()
        logger.debug(
            LogRecord(
                event=LogEvent.OPENAI_RESPONSE.value,
                message="OpenAI response received",
                request_id=request_id,
                data={"response": response_dict},
            )
        )

        if "error" in response_dict and response_dict["error"] is not None:
            error_details = response_dict["error"]
            error_code = error_details.get("code", 500)
            error_msg = error_details.get("message", "Unknown provider error")

            provider_details = None
            if isinstance(error_details, dict) and "metadata" in error_details:
                provider_details = extract_provider_error_details(error_details)

            error_type = STATUS_CODE_ERROR_MAP.get(
                error_code, models.AnthropicErrorType.API_ERROR
            )

            if error_code == 429:
                error_type = models.AnthropicErrorType.RATE_LIMIT

            return await _handle_error(
                request=request,
                status_code=error_code,
                anthropic_error_type=error_type,
                error_message=error_msg,
                provider_details=provider_details,
            )

        anthropic_response = conversion.convert_openai_to_anthropic(
            openai_response, anthropic_request.model, request_id=request_id
        )
        usage_info = anthropic_response.usage.model_dump()

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            LogRecord(
                event=LogEvent.REQUEST_COMPLETED.value,
                message="Request completed successfully",
                request_id=request_id,
                data={
                    "status_code": 200,
                    "duration_ms": duration_ms,
                    "input_tokens": usage_info.get("input_tokens"),
                    "output_tokens": usage_info.get("output_tokens"),
                },
            )
        )

        logger.debug(
            LogRecord(
                event=LogEvent.ANTHROPIC_RESPONSE.value,
                message="Anthropic response prepared",
                request_id=request_id,
                data={"response": anthropic_response.model_dump(exclude_unset=True)},
            )
        )

        return JSONResponse(content=anthropic_response.model_dump(exclude_unset=True))


@app.post(
    "/v1/messages/count_tokens",
    response_model=models.TokenCountResponse,
    tags=["Utility"],
)
async def count_tokens_endpoint(request: Request) -> models.TokenCountResponse:
    """
    Counts tokens for the given messages and system prompt using tiktoken.
    """
    request_id = str(uuid.uuid4()).split("-")[0]
    request.state.request_id = request_id
    start_time = time.time()

    body = await request.json()

    request_data = models.TokenCountRequest.model_validate(body)

    token_count = token_counter.count_tokens_for_request(
        messages=request_data.messages,
        system=request_data.system,
        model_name=request_data.model,
        tools=request_data.tools,
        request_id=request_id,
    )

    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        LogRecord(
            event=LogEvent.TOKEN_COUNT.value,
            message=f"Counted {token_count} tokens for request",
            request_id=request_id,
            data={
                "duration_ms": duration_ms,
                "token_count": token_count,
                "model": request_data.model,
            },
        )
    )
    return models.TokenCountResponse(input_tokens=token_count)


@app.get("/", include_in_schema=False, tags=["Health"])
async def root() -> JSONResponse:
    """Basic health check and information endpoint."""
    logger.debug(
        LogRecord(
            event=LogEvent.HEALTH_CHECK.value,
            message="Root health check endpoint accessed",
        )
    )
    return JSONResponse(
        {"proxy": settings.app_name, "version": settings.app_version, "status": "ok"}
    )


@app.get("/health", include_in_schema=False, tags=["Health"])
async def health() -> JSONResponse:
    """Basic health check and information endpoint."""
    logger.debug(
        LogRecord(
            event=LogEvent.HEALTH_CHECK.value,
            message="Root health check endpoint accessed",
        )
    )
    return JSONResponse(
        {"proxy": settings.app_name, "version": settings.app_version, "status": "ok"}
    )


@app.get("/status", include_in_schema=False, tags=["Status"])
async def status() -> JSONResponse:
    """Basic health check and information endpoint."""
    logger.debug(
        LogRecord(
            event=LogEvent.HEALTH_CHECK.value,
            message="Root health check endpoint accessed",
        )
    )
    return JSONResponse(
        {"proxy": settings.app_name, "version": settings.app_version, "status": "ok"}
    )


def get_error_response(
    error_type: models.AnthropicErrorType,
    message: str,
    provider_details: Optional[models.ProviderErrorMetadata] = None,
) -> models.AnthropicErrorResponse:
    """
    Formats error response in Anthropic-compatible structure with optional provider details.

    Creates a consistent error response that follows the Anthropic API error format,
    but with enhanced provider-specific error details when available. This allows
    clients to get detailed information about errors from underlying providers.

    Examples:
        Basic error without provider details:
        ```python
        get_error_response(
            models.AnthropicErrorType.INVALID_REQUEST,
            "Invalid parameter value"
        )
        ```

        Error with provider details:
        ```python
        provider_details = models.ProviderErrorMetadata(
            provider_name="google",
            raw_error={"error": {"code": 400, "message": "Invalid BatchTool schema"}}
        )
        get_error_response(
            models.AnthropicErrorType.INVALID_REQUEST,
            "Provider returned error",
            provider_details
        )
        ```

    Args:
        error_type: The Anthropic error type enum
        message: The error message
        provider_details: Optional provider-specific error details

    Returns:
        A fully formatted Anthropic-style error response
    """
    error = models.AnthropicErrorDetail(type=error_type.value, message=message)

    if provider_details:
        error.provider = provider_details.provider_name

        if (
            provider_details.raw_error
            and isinstance(provider_details.raw_error, dict)
            and "error" in provider_details.raw_error
        ):
            provider_error = provider_details.raw_error["error"]
            if isinstance(provider_error, dict):
                if "message" in provider_error and provider_error["message"]:
                    error.provider_message = provider_error["message"]

                if "code" in provider_error:
                    error.provider_code = provider_error["code"]

    return models.AnthropicErrorResponse(error=error)


async def _handle_error(
    request: Request,
    status_code: int,
    anthropic_error_type: models.AnthropicErrorType,
    error_message: str,
    provider_details: Optional[models.ProviderErrorMetadata] = None,
    exc: Optional[Exception] = None,
) -> JSONResponse:
    """
    Centralized logic for logging and formatting API error responses.

    This function handles all error responses consistently by:
    1. Using the provided error message and details
    2. Logging the error with appropriate context
    3. Creating a properly formatted Anthropic error response

    Args:
        request: The FastAPI request object
        status_code: HTTP status code to return
        anthropic_error_type: The Anthropic error type enum value
        error_message: The error message to include in the response
        provider_details: Optional provider-specific error details
        exc: Optional exception that was caught (for logging)

    Returns:
        A JSON response with properly formatted error details
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()).split("-")[0])

    duration = (
        (time.time() - request.state.start_time) * 1000
        if hasattr(request.state, "start_time")
        else 0
    )

    log_data = {
        "status_code": status_code,
        "duration_ms": duration,
        "error_type": anthropic_error_type.value,
    }

    if provider_details:
        log_data["provider"] = provider_details.provider_name
        log_data["provider_error"] = provider_details.raw_error

    record = LogRecord(
        event=LogEvent.REQUEST_FAILURE.value,
        message=f"Request error: {error_message}",
        request_id=request_id,
        data=log_data,
    )
    logger.error(record, exc=exc)

    error_response = get_error_response(
        anthropic_error_type, error_message, provider_details
    )

    return JSONResponse(status_code=status_code, content=error_response.model_dump())


async def _handle_exception(
    request: Request,
    exc: Exception,
    status_code: int,
    anthropic_error_type: models.AnthropicErrorType,
) -> JSONResponse:
    """
    Handles errors from exceptions by extracting error details and delegating to _handle_error.

    Args:
        request: The FastAPI request object
        exc: The exception that was caught
        status_code: HTTP status code to return
        anthropic_error_type: The Anthropic error type enum value

    Returns:
        A JSON response with properly formatted error details
    """
    detail = str(exc)

    provider_details = None
    if hasattr(exc, "body") and isinstance(exc.body, dict):
        error_details = exc.body.get("error", {})
        if isinstance(error_details, dict):
            provider_details = extract_provider_error_details(error_details)

    return await _handle_error(
        request=request,
        status_code=status_code,
        anthropic_error_type=anthropic_error_type,
        error_message=detail,
        provider_details=provider_details,
        exc=exc,
    )


@app.exception_handler(ValidationError)
async def validation_exception_handler(
    request: Request, exc: ValidationError
) -> JSONResponse:
    """Handles Pydantic validation errors (422)."""
    return await _handle_exception(
        request, exc, 422, models.AnthropicErrorType.INVALID_REQUEST
    )


@app.exception_handler(json.JSONDecodeError)
async def json_decode_exception_handler(
    request: Request, exc: json.JSONDecodeError
) -> JSONResponse:
    """Handles JSON parsing errors (400)."""
    return await _handle_exception(
        request, exc, 400, models.AnthropicErrorType.INVALID_REQUEST
    )


@app.exception_handler(openai.APIError)
async def openai_api_error_handler(
    request: Request, exc: openai.APIError
) -> JSONResponse:
    """
    Handles all OpenAI API errors, mapping them to appropriate Anthropic types.

    Maps specific OpenAI/OpenRouter exception types to the proper Anthropic error types
    using a consistent mapping strategy. Ensures correct status codes are returned
    based on the exception type.
    """
    status_code = 500
    anthropic_error_type = models.AnthropicErrorType.API_ERROR

    if isinstance(exc, openai.AuthenticationError):
        status_code = 401
        anthropic_error_type = models.AnthropicErrorType.AUTHENTICATION
    elif isinstance(exc, openai.RateLimitError):
        status_code = 429
        anthropic_error_type = models.AnthropicErrorType.RATE_LIMIT
    elif isinstance(exc, openai.BadRequestError):
        status_code = 400
        anthropic_error_type = models.AnthropicErrorType.INVALID_REQUEST
    elif isinstance(exc, openai.PermissionDeniedError):
        status_code = 403
        anthropic_error_type = models.AnthropicErrorType.PERMISSION
    elif isinstance(exc, openai.NotFoundError):
        status_code = 404
        anthropic_error_type = models.AnthropicErrorType.NOT_FOUND
    elif isinstance(exc, openai.UnprocessableEntityError):
        status_code = 422
        anthropic_error_type = models.AnthropicErrorType.INVALID_REQUEST
    elif isinstance(exc, openai.InternalServerError):
        status_code = 500
        anthropic_error_type = models.AnthropicErrorType.API_ERROR
    elif isinstance(exc, openai.APIConnectionError):
        status_code = 502
        anthropic_error_type = models.AnthropicErrorType.API_ERROR
    elif isinstance(exc, openai.APITimeoutError):
        status_code = 504
        anthropic_error_type = models.AnthropicErrorType.API_ERROR
    elif isinstance(exc, openai.APIStatusError):
        status_code = exc.status_code if hasattr(exc, "status_code") else 500

        default_error = (
            models.AnthropicErrorType.API_ERROR
            if status_code >= 500
            else models.AnthropicErrorType.INVALID_REQUEST
        )

        anthropic_error_type = STATUS_CODE_ERROR_MAP.get(status_code, default_error)

    return await _handle_exception(request, exc, status_code, anthropic_error_type)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handles any other unexpected errors (500).

    This is the catch-all handler for any exceptions not specifically handled
    by other exception handlers. It always returns a 500 API error.
    """
    return await _handle_exception(
        request, exc, 500, models.AnthropicErrorType.API_ERROR
    )


@app.middleware("http")
async def add_process_time_header(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    request.state.start_time = time.time()
    response = await call_next(request)
    return response
