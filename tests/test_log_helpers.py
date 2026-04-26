import logging
import sys

sys.path.insert(0, "src")

from main import (
    LogEvent,
    LogRecord,
    PrettyConsoleFormatter,
    mask_secrets,
)


# ---------------------------------------------------------------------------
# mask_secrets
# ---------------------------------------------------------------------------


class TestMaskSecrets:
    def test_long_authorization_header(self):
        headers = {
            "authorization": "OAuth y1__xCapOORpdT-ARiuKyCqndUCxxxxxxxxxxxxxxxxxxxxxxx",
            "content-type": "application/json",
        }
        masked = mask_secrets(headers)
        # First 8 chars of "OAuth y1__xCapOORpdT-..." are "OAuth y1"
        assert masked["authorization"] == "OAuth y1...XXX"
        assert masked["content-type"] == "application/json"

    def test_short_value(self):
        headers = {"x-api-key": "short"}
        masked = mask_secrets(headers)
        assert masked["x-api-key"] == "short...XXX"

    def test_no_secrets(self):
        headers = {"content-type": "application/json", "accept": "application/json"}
        masked = mask_secrets(headers)
        assert masked == headers

    def test_case_insensitive(self):
        headers = {"Authorization": "Bearer sk-1234567890abcdef"}
        masked = mask_secrets(headers)
        # First 8 chars of "Bearer sk-1234567890abcdef" are "Bearer s"
        assert masked["Authorization"] == "Bearer s...XXX"

    def test_x_api_key(self):
        headers = {"x-api-key": "sk-ant-api03-verylongkey"}
        masked = mask_secrets(headers)
        assert masked["x-api-key"] == "sk-ant-a...XXX"

    def test_api_key_header(self):
        headers = {"api-key": "1234567890abcdef"}
        masked = mask_secrets(headers)
        assert masked["api-key"] == "12345678...XXX"

    def test_cookie_header(self):
        headers = {"cookie": "session=abc123def456ghi789"}
        masked = mask_secrets(headers)
        # First 8 chars: "session="
        assert masked["cookie"] == "session=...XXX"

    def test_set_cookie_header(self):
        headers = {"set-cookie": "token=xyz987longvalue"}
        masked = mask_secrets(headers)
        assert masked["set-cookie"] == "token=xy...XXX"

    def test_proxy_authorization_header(self):
        headers = {"proxy-authorization": "Basic dXNlcjpwYXNz"}
        masked = mask_secrets(headers)
        assert masked["proxy-authorization"] == "Basic dX...XXX"

    def test_exactly_8_chars(self):
        """Values of exactly 8 chars are > 8 is False, so short path."""
        headers = {"authorization": "12345678"}
        masked = mask_secrets(headers)
        # len("12345678") == 8, which is NOT > 8, so short path
        assert masked["authorization"] == "12345678...XXX"

    def test_9_chars_long_path(self):
        """Values of 9 chars trigger the long path (> 8)."""
        headers = {"authorization": "123456789"}
        masked = mask_secrets(headers)
        assert masked["authorization"] == "12345678...XXX"

    def test_empty_value(self):
        headers = {"authorization": ""}
        masked = mask_secrets(headers)
        # len("") == 0, not > 8, so short path
        assert masked["authorization"] == "...XXX"

    def test_preserves_key_case(self):
        """Original key casing should be preserved."""
        headers = {"X-Api-Key": "longapikeyvalue123"}
        masked = mask_secrets(headers)
        assert "X-Api-Key" in masked
        # First 8 chars of "longapikeyvalue123": "longapik"
        assert masked["X-Api-Key"] == "longapik...XXX"


# ---------------------------------------------------------------------------
# PrettyConsoleFormatter
# ---------------------------------------------------------------------------


def _make_log_record(event: str, message: str, request_id=None, data=None):
    """Helper to create a LogRecord (our custom dataclass)."""
    return LogRecord(
        event=event,
        message=message,
        request_id=request_id,
        data=data,
    )


def _make_python_log_record(log_record: LogRecord, level=logging.DEBUG):
    """Create a stdlib logging.LogRecord with our custom LogRecord attached."""
    record = logging.LogRecord(
        name="AnthropicProxy",
        level=level,
        pathname="test",
        lineno=0,
        msg=log_record.message,
        args=(),
        exc_info=None,
    )
    record.log_record = log_record
    return record


class TestPrettyConsoleFormatterNonStructured:
    def test_plain_message(self):
        formatter = PrettyConsoleFormatter()
        py_record = logging.LogRecord(
            name="AnthropicProxy",
            level=logging.INFO,
            pathname="test",
            lineno=0,
            msg="Simple log message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(py_record)
        assert "Simple log message" in result
        assert "INF" in result

    def test_uvicorn_message(self):
        formatter = PrettyConsoleFormatter()
        py_record = logging.LogRecord(
            name="uvicorn",
            level=logging.INFO,
            pathname="test",
            lineno=0,
            msg="Application startup complete",
            args=(),
            exc_info=None,
        )
        result = formatter.format(py_record)
        assert "Application startup complete" in result


class TestPrettyConsoleFormatterFallback:
    def test_unknown_event_with_data(self):
        formatter = PrettyConsoleFormatter()
        lr = _make_log_record(
            event="unknown_event",
            message="Something happened",
            data={"key1": "val1", "key2": "val2"},
        )
        py_record = _make_python_log_record(lr)
        result = formatter.format(py_record)
        assert "Something happened" in result
