import argparse
import tkinter as tk
from tkinter import filedialog
from train_utils import *
from predict_utils import *

def main():

    def get_file_path():
        # Create a root window but keep it hidden
        root = tk.Tk()
        root.withdraw()
    
        # Open a file dialog and get the file path
        file_path = filedialog.askopenfilename()
        return file_path
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Predict the class of an image using a trained model")
    parser.add_argument("--file", type=str, help="Path to the file")
    # parser.add_argument('input_image', type=str, help='Path to the image')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint')
    parser.add_argument('--top_k', type=int, default=3, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Use a mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

    args, _ = parser.parse_known_args()
    if not args.file:
        args.file = get_file_path()
    
    device = init_device(args.gpu)
    
    model, optimizer = load_checkpoint(args.checkpoint)
    model.to(device)
    
    cat_to_name = init_dict(args.category_names)

    # imshow(process_image(args.file)

    display_prediction(args.file, model, device, cat_to_name, args.top_k)

if __name__ == '__main__':
    main()

# Example usage
# python predict.py checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu
