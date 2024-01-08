import argparse
import tkinter as tk
from tkinter import filedialog

def get_file_path():
    # Create a root window but keep it hidden
    root = tk.Tk()
    root.withdraw()

    # Open a file dialog and get the file path
    file_path = filedialog.askopenfilename()
    return file_path

# Set up argparse
parser = argparse.ArgumentParser(description="Your script description")
parser.add_argument("--file", type=str, help="Path to the file")

# Parse known arguments, leaving any unknown ones
args, _ = parser.parse_known_args()

# If file argument is not provided, open file chooser
if not args.file:
    args.file = get_file_path()

# Rest of your script
print(f"File chosen: {args.file}")
