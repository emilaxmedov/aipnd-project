import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import json

from datetime import datetime
        
def init_datasets(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, data_transforms['test'])
    }
    
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
    }
    return data_transforms, image_datasets, dataloaders

def init_dict(category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def init_device(gpu):
    if gpu == True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    return device

def init_variables(arch, epochs, learning_rate, hidden_units):
    l_arch = arch
    l_epochs = epochs
    print_every = 10
    l_learning_rate = learning_rate
    
    l_hidden_units = hidden_units
    output_features = 102
    dropout_p = 0.2
    return l_arch, l_epochs, print_every, l_learning_rate, l_hidden_units, output_features, dropout_p

def build_classifier(input_features, hidden_units, output_features, dropout_p=0.5):
    # Build a feed-forward network as classifier
    classifier = nn.Sequential()
    layer_sizes = [input_features] + hidden_units
    for i in range(len(layer_sizes)-1):
        classifier.add_module('fc{}'.format(i), nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        classifier.add_module('relu{}'.format(i), nn.ReLU())
        classifier.add_module('drop{}'.format(i), nn.Dropout(p=dropout_p))
    classifier.add_module('output', nn.Linear(layer_sizes[-1], output_features))
    classifier.add_module('softmax', nn.LogSoftmax(dim=1))
    
    return classifier

def build_model(arch, hidden_units, output_features, dropout_p, learning_rate):
    model = getattr(models, arch)(pretrained=True)

    input_features = 0
    
    if isinstance(model, models.vgg.VGG):
        # For VGG models
        input_features = model.classifier[0].in_features
    elif isinstance(model, models.densenet.DenseNet):
        # For models like DenseNet
        input_features = model.classifier.in_features
    else:
        raise Exception("Unknown model type for building classifier")
    
    classifier = build_classifier(input_features, hidden_units, output_features, dropout_p)
    
    for param in model.features.parameters():
        param.requires_grad = False
    
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    return model, criterion, optimizer, input_features

def validate_model(device, model, dataloader, criterion, running_loss, print_every):
    model.eval()
    valid_loss = 0
    accuracy = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs)
            batch_loss = criterion(logps, labels)

            valid_loss += batch_loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    model.train()
    
    print(f"Training loss: {running_loss / print_every:.3f}.. "
          f"Validation loss: {valid_loss / len(dataloader):.3f}.. "
          f"Validation accuracy: {accuracy / len(dataloader)*100:.3f}%")

def train_model(device, model, dataloaders, criterion, optimizer, epochs, print_every):
    model.to(device)
    steps = 0
    start_time = datetime.now()

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if steps % print_every == 0:
                validate_model(device, model, dataloaders['valid'], criterion, running_loss, print_every)

        print(f"Epoch {epoch+1}/{epochs} completed.")
    
        
    print('Training duration: {}'.format(datetime.now() - start_time))

def test_network(device, model, dataloaders):
    start_time = datetime.now()
    model.eval()
    accuracy = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
    
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    accuracy_percentage = 100 * accuracy / total
    
    print(f'Accuracy of the network on the test images: {accuracy_percentage:.2f}%')
    print('test duration: {}'.format(datetime.now() - start_time))

def save_checkpoint(model, optimizer, save_path, epochs, arch, input_features, image_datasets, hidden_units, output_features, dropout_p):
    checkpoint = {
        'arch': arch,
        'state_dict': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'class_to_idx': image_datasets['train'].class_to_idx,
        'num_epochs': epochs,
        'input_features': input_features,
        'hidden_units': hidden_units,
        'output_features': output_features,
        'dropout_p': dropout_p
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    model = getattr(models, checkpoint['arch'])(pretrained=True)
    
    model.classifier = build_classifier(checkpoint['input_features'], checkpoint['hidden_units'], checkpoint['output_features'], checkpoint['dropout_p'])
    
    model.load_state_dict(checkpoint['state_dict'])
    
    # Rebuild the optimizer
    optimizer = torch.optim.Adam(model.classifier.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model, optimizer