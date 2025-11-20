# src/utils.py
import os
import sys
import subprocess
import platform
import shutil
from datetime import datetime

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

def start_comfyui(script_path):
    """Starts the ComfyUI server using the provided script path, ensuring the correct working directory."""
    if not script_path or not os.path.exists(script_path):
        print(f"\n{Colors.RED}ERROR: ComfyUI startup script not found or not configured.{Colors.ENDC}")
        print(f"  -> Please check the 'startup_script' value in your global_config.json.")
        return

    # Get the directory containing the script. This is the crucial change.
    script_dir = os.path.dirname(script_path)

    print(f"Attempting to start ComfyUI from directory: {script_dir}")
    try:
        if platform.system() == "Windows":
            # Pass the script directory to the `cwd` argument.
            subprocess.Popen([script_path], creationflags=subprocess.CREATE_NEW_CONSOLE, cwd=script_dir)
            print(f"{Colors.GREEN}  -> ComfyUI should be starting in a new console window.{Colors.ENDC}")
        else:
            # Also apply the cwd fix for other platforms.
            subprocess.Popen([script_path], start_new_session=True, cwd=script_dir)
            print(f"{Colors.GREEN}  -> ComfyUI is starting as a background process.{Colors.ENDC}")
            print(f"{Colors.YELLOW}     (You may need to monitor its console output manually){Colors.ENDC}")
            
    except Exception as e:
        print(f"{Colors.RED}  -> ERROR: Failed to start ComfyUI script. {e}{Colors.ENDC}")

def archive_folder(base_path, folder_name):
    """
    Renames a folder to folder_name_backup_TIMESTAMP.
    Returns True if archived, False if folder didn't exist.
    """
    target_dir = os.path.join(base_path, folder_name)
    if os.path.exists(target_dir) and os.listdir(target_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = f"{folder_name}_backup_{timestamp}"
        new_path = os.path.join(base_path, new_name)
        try:
            os.rename(target_dir, new_path)
            print(f"  -> Archived existing '{folder_name}' to '{new_name}'")
            return True
        except OSError as e:
            print(f"  -> {Colors.RED}ERROR: Could not archive folder: {e}{Colors.ENDC}")
            return False
    return False
