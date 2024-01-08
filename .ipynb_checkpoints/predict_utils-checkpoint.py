import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from PIL import Image

import torch

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image)
    
    if image.width < image.height:
        size = (256, int((256 / image.width) * image.height))
    else:
        size = (int((256 / image.height) * image.width), 256)
    
    image = image.resize(size, Image.Resampling.LANCZOS)

    # Center crop the image to 224 x 224
    width, height = image.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2

    image = image.crop((left, top, right, bottom))

    np_image = np.array(image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    np_image = np_image.transpose((2, 0, 1))

    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    img = process_image(image_path)
    
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze_(0).to(device)
    
    with torch.no_grad():
        model.eval()
        model.to(device)
        out = model(img_tensor)
    ps = torch.exp(out)
    
    topk, topk_idx = ps.topk(topk, dim=1)
    
    topk = topk.detach().cpu().numpy().tolist()[0]  
    topk_idx = topk_idx.detach().cpu().numpy().tolist()[0]
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    topk_classes = [idx_to_class[index] for index in topk_idx]
    
    return topk, topk_classes

# TODO: Display an image along with the top 5 classes
def display_prediction(image_path, model, device, cat_to_name, topk):
    probs, classes = predict(image_path, model, device, topk)
    print (probs)
    print(classes)
    classes = [cat_to_name[str(cls)] for cls in classes]
    
    plt.figure(figsize=(6,10))
    ax = plt.subplot(2,1,1)
    
    img = process_image(image_path)
    imshow(img, ax, title=classes[0])
    
    plt.subplot(2,1,2)
    sns.barplot(x=probs, y=classes, color=sns.color_palette()[0])
    plt.show()