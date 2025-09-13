# src/utils.py
import os
import sys
import subprocess

# A simple class to hold ANSI color codes for terminal output
class Colors:
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m' # Resets color to default

def get_char():
    """Gets a single character from the user without waiting for Enter."""
    if sys.platform == "win32":
        import msvcrt
        return msvcrt.getch().decode('utf-8')
    else:
        import tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

def get_menu_choice():
    """Gets a single character from the user and returns it in lowercase."""
    char = get_char()
    print(char) # Echo the character back to the user
    return char.lower()

def open_folder_in_explorer(path):
    """Opens the given folder path in the default file explorer."""
    print(f"Opening folder: {path}")
    try:
        if sys.platform == "win32":
            os.startfile(os.path.realpath(path))
        elif sys.platform == "darwin":
            subprocess.call(["open", path])
        else:
            subprocess.call(["xdg-open", path])
    except Exception as e:
        print(f"  -> ERROR: Could not open folder. {e}")

# NEW FUNCTION to open a specific file
def open_file(file_path):
    """Opens a specific file in the default application."""
    print(f"Opening file: {file_path}")
    try:
        if sys.platform == "win32":
            os.startfile(os.path.realpath(file_path))
        elif sys.platform == "darwin":
            subprocess.call(["open", file_path])
        else:
            subprocess.call(["xdg-open", file_path])
    except Exception as e:
        print(f"  -> ERROR: Could not open file. {e}")
