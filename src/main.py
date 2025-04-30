"""
Main entry point for running the Claude Proxy application using Uvicorn.
Displays a startup banner with configuration details.
"""

import uvicorn
import signal
import sys
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from claude_proxy import logger
from claude_proxy.config import settings
from claude_proxy.logger import console


def handle_exit_signal(signal_received, frame):
    print("\nGracefully shutting down...")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_exit_signal)
    signal.signal(signal.SIGTERM, handle_exit_signal)

    console.print(
        r"""[bold blue]
           /$$                           /$$
          | $$                          | $$
  /$$$$$$$| $$  /$$$$$$  /$$   /$$  /$$$$$$$  /$$$$$$         /$$$$$$   /$$$$$$   /$$$$$$  /$$   /$$ /$$   /$$
 /$$_____/| $$ |____  $$| $$  | $$ /$$__  $$ /$$__  $$       /$$__  $$ /$$__  $$ /$$__  $$|  $$ /$$/| $$  | $$
| $$      | $$  /$$$$$$$| $$  | $$| $$  | $$| $$$$$$$$      | $$  \ $$| $$  \__/| $$  \ $$ \  $$$$/ | $$  | $$
| $$      | $$ /$$__  $$| $$  | $$| $$  | $$| $$_____/      | $$  | $$| $$      | $$  | $$  >$$  $$ | $$  | $$
|  $$$$$$$| $$|  $$$$$$$|  $$$$$$/|  $$$$$$$|  $$$$$$$      | $$$$$$$/| $$      |  $$$$$$/ /$$/\  $$|  $$$$$$$
 \_______/|__/ \_______/ \______/  \_______/ \_______/      | $$____/ |__/       \______/ |__/  \__/ \____  $$
                                                            | $$                                     /$$  | $$
                                                            | $$                                    |  $$$$$$/
                                                            |__/                                     \______/ 
    [/]""",
        justify="left",
    )

    config_details = Text.assemble(
        ("   Version       : ", "default"),
        (f"v{settings.app_version}", "bold red"),
        ("\n   Big Model     : ", "default"),
        (settings.big_model_name, "magenta"),
        ("\n   Small Model   : ", "default"),
        (settings.small_model_name, "cyan"),
        ("\n   Log Level     : ", "default"),
        (settings.log_level, "yellow"),
        ("\n   Listening on  : ", "default"),
        (f"http://{settings.host}:{settings.port}", "bold green"),
        ("\n   Reload        : ", "default"),
        ("Enabled", "bold orange1") if settings.reload else ("Disabled", "dim"),
    )
    console.print(
        Panel(
            config_details,
            title="Configuration",
            border_style="blue",
            title_align="left",
            expand=False,
        )
    )
    console.print("\n\n")
    console.print(Rule("Starting uvicorn server...", style="dim blue"))

    try:
        uvicorn.run(
            app="claude_proxy.api:app",
            host=settings.host,
            port=settings.port,
            reload=settings.reload,
            log_config=None,
            access_log=True,
        )
    except Exception as e:
        logger._logger.error(f"Failed to start Uvicorn server: {e}", e)
        exit(1)
