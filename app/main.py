# OVAKO SE POKREĆE:
# python3 -m app.main
# N.M.

import itertools
import os
import shutil
import sys
import textwrap
import threading
import time

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.formatted_text import ANSI

from brain.core import CIPHERBrain


BLUE = "\033[94m"
CYAN = "\033[96m"
WHITE = "\033[97m"
GRAY = "\033[90m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

WIDTH = 90

LINE = f"{BLUE}{'=' * WIDTH}{RESET}"
THIN_LINE = f"{GRAY}{'─' * WIDTH}{RESET}"

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class Spinner:
    """Simple animated terminal spinner that runs on a background thread
    while a blocking call (search + fetch + synthesize) is in progress.
    A true progress bar isn't possible here since we don't know up front
    how many pages will end up being fetched — this gives the same
    reassurance ('something is happening') without faking a percentage."""

    def __init__(self, message: str = "Thinking"):
        self.message = message
        self._stop_event = threading.Event()
        self._thread = None

    def _spin(self):
        frames = itertools.cycle(SPINNER_FRAMES)

        while not self._stop_event.is_set():
            frame = next(frames)
            line = f"{CYAN}{frame} {self.message}...{RESET}"
            sys.stdout.write(f"\r{line}")
            sys.stdout.flush()
            time.sleep(0.08)

        clear_width = len(self.message) + 12
        sys.stdout.write("\r" + " " * clear_width + "\r")
        sys.stdout.flush()

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

        if self._thread:
            self._thread.join()


def clear_screen():
    os.system("clear")


def print_banner():
    logo = """
   █████████  █████ ███████████  █████   █████ ██████████ ███████████
  ███░░░░░███░░███ ░░███ ░░███░░░░░███░░███   ░░███ ░░███ ░░███░░░░░███
 ███     ░░░  ░███  ░███    ░███ ░███    ░███  ░███  █ ░  ░███    ░███
░███          ░███  ░██████████  ░███████████  ░██████    ░██████████
░███          ░███  ░███░░░░░░   ░███░░░░░███  ░███░░█    ░███░░░░░███
░░███     ███ ░███  ░███         ░███    ░███  ░███ ░   █ ░███    ░███
 ░░█████████  █████ █████        █████   █████ ██████████ █████   █████
  ░░░░░░░░░  ░░░░░ ░░░░░        ░░░░░   ░░░░░░ ░░░░░░░░░░ ░░░░░   ░░░░░
"""

    print()
    print(LINE)
    print()

    for line in logo.splitlines():
        if line.strip():
            print(f"{BLUE}{line.center(WIDTH)}{RESET}")

    print()
    print(f"{CYAN}{'Your personal AI assistant':^{WIDTH}}{RESET}")
    print()
    print(LINE)
    print()


def wrap_width():
    terminal_width = shutil.get_terminal_size(fallback=(WIDTH, 24)).columns
    return max(40, min(terminal_width - 4, WIDTH))


def print_response(response):
    print()

    print(
        f"{BLUE}{BOLD}"
        f"● CIPHER"
        f"{RESET}"
    )

    print(f"{GRAY}{'─' * 12}{RESET}")
    print()

    width = wrap_width()

    paragraphs = [
        paragraph.strip()
        for paragraph in response.split("\n\n")
        if paragraph.strip()
    ]

    for index, paragraph in enumerate(paragraphs):
        wrapped = textwrap.fill(
            paragraph,
            width=width,
            replace_whitespace=True
        )

        print(f"{WHITE}{wrapped}{RESET}")

        if index != len(paragraphs) - 1:
            print()

    print()
    print(THIN_LINE)
    print()


def print_error(message, error=None):
    print()

    print(
        f"{RED}{BOLD}"
        f"✖ ERROR"
        f"{RESET}"
    )

    print(f"{RED}{message}{RESET}")

    if error:
        print()
        print(f"{GRAY}{error}{RESET}")

    print()
    print(THIN_LINE)
    print()


def print_goodbye():
    print()

    print(
        f"{BLUE}{BOLD}"
        f"● CIPHER"
        f"{RESET}"
    )

    print(f"{GRAY}{'─' * 12}{RESET}")

    print(
        f"{CYAN}Session terminated.{RESET}"
    )

    print(
        f"{GRAY}Goodbye.{RESET}"
    )

    print()
    print(LINE)
    print()


def main():
    clear_screen()
    print_banner()

    message = "Type anything to talk with CIPHER."
    exit_message = "Type 'exit' to quit."

    print(
        f"{GRAY}{message:^{WIDTH}}{RESET}"
    )

    print(
        f"{GRAY}{exit_message:^{WIDTH}}{RESET}"
    )

    print()

    print(
        f"{GREEN}{BOLD}"
        f"{'● CIPHER ONLINE':^{WIDTH}}"
        f"{RESET}"
    )

    print()

    brain = CIPHERBrain()

    history = InMemoryHistory()

    session = PromptSession(
        history=history
    )

    while True:
        try:
            prompt = ANSI(
                f"{CYAN}{BOLD}"
                f"You"
                f"{RESET} "
                f"{BLUE}❯{RESET} "
            )

            user_input = session.prompt(
                prompt
            ).strip()

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit"}:
                print_goodbye()
                break

            spinner = Spinner("CIPHER is thinking")
            spinner.start()

            try:
                response = brain.think(user_input)
            finally:
                spinner.stop()

            print_response(response.answer)

        except KeyboardInterrupt:
            print()
            continue

        except EOFError:
            print_goodbye()
            break

        except Exception as error:
            print_error(
                "CIPHER encountered an unexpected error.",
                error
            )


if __name__ == "__main__":
    main()