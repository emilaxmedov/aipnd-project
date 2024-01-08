import argparse
from train_utils import *

def main():

    def parse_hidden_units(string):
        try:
            return [int(item) for item in string.split(',')]
        except ValueError:
            raise argparse.ArgumentTypeError('Hidden units must be a comma-separated list of integers')
            
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset")
    parser.add_argument('data_directory', type=str, help='Directory of the dataset')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=parse_hidden_units, default='4096,1000', help='List of hidden units. For VGG16 - 4096,1000, for DenSenet121 - 512,256')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    args = parser.parse_args()

    data_transforms, image_datasets, dataloaders = init_datasets(args.data_directory)

    cat_to_name = init_dict('cat_to_name.json')
    
    device = init_device(args.gpu)

    arch, epochs, print_every, learning_rate, hidden_units, output_features, dropout_p = init_variables(args.arch, args.epochs, args.learning_rate, args.hidden_units)

    model, criterion, optimizer, input_features = build_model(args.arch, args.hidden_units, output_features, dropout_p, args.learning_rate)

    train_model(device, model, dataloaders, criterion, optimizer, epochs, print_every)

    test_network(device, model, dataloaders)

    save_checkpoint(model, optimizer, args.save_dir, epochs, arch, input_features, image_datasets, hidden_units, output_features, dropout_p)

if __name__ == '__main__':
    main()

# Example usage
# python train.py flowers --save_dir checkpoint.pth --arch "densenet121" --learning_rate 0.001 --hidden_units 512,256 --epochs 5 --gpu
