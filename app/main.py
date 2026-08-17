import os

BLUE = "\033[94m"
CYAN = "\033[96m"
WHITE = "\033[97m"
GRAY = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"

WIDTH = 90

LINE = f"{BLUE}{'=' * WIDTH}{RESET}"
THIN_LINE = f"{GRAY}{'-' * WIDTH}{RESET}"

def clear_screen():
    os.system("clear")

def print_banner():
    logo = """
   █████████  █████ ███████████  █████   █████ ██████████ ███████████
  ███░░░░░███░░███ ░░███░░░░░███░░███   ░░███ ░░███░░░░░█░░███░░░░░███
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

# Main

def main():
    clear_screen()
    print_banner()

    message = "Type anything to talk with CIPHER."
    exit_message = "Type 'exit' to quit."

    print(f"{GRAY}{message:^{WIDTH}}{RESET}")
    print(f"{GRAY}{exit_message:^{WIDTH}}{RESET}")
    print()

    while True:
        try:
            user_input = input(
                f"{CYAN}{BOLD}You{RESET} {BLUE}❯{RESET} "
            ).strip()

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit"}:
                print()
                print(f"{BLUE}{BOLD}CIPHER{RESET}: Goodbye!")
                print()
                break

            # SAMO TRENUTACNO
            # KASNIJE ĆE SE OVO POZIVATI NA brain/core.py 
            # N.M.
            
            print()
            print(f"{BLUE}{BOLD}CIPHER{RESET}: {WHITE}{user_input}{RESET}")
            print()
            print(THIN_LINE)
            print()

        except KeyboardInterrupt:
            print()
            print()
            print(f"{BLUE}{BOLD}CIPHER{RESET}: Goodbye!")
            print()
            break

        except EOFError:
            print()
            print()
            print(f"{BLUE}{BOLD}CIPHER{RESET}: Goodbye!")
            print()
            break


if __name__ == "__main__":
    main()