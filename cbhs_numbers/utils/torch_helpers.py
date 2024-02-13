
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Digit_OCR_CNN(nn.Module):

    def __init__(self):
        super(Digit_OCR_CNN, self).__init__()
        # Convolutional layers learn to identify the key features of an input image.
        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)
        self.conv2 = nn.Conv2d(3, 6, 3, padding=1)
        # Pooling layers reduce the number of neurons.
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers classify the convolved image into a final output.
        self.fc1 = nn.Linear(6 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 6 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x
    
class Digit_OCR_CNN2(nn.Module):

    def __init__(self):
        super(Digit_OCR_CNN2, self).__init__()


        self.fc1 = nn.Linear(28*28, 128)
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, state):
        x = state.view(-1, 28*28)

        x = F.relu(self.fc1(x))
        x = self.drop1(x)

        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)

        return x

def load_model(weights_path: str = 'digit_mnist_model.pt', relative=True):
    if relative:
        weights_path = os.path.join(os.path.dirname(__file__), weights_path)
    net = Digit_OCR_CNN()
    net.eval()
    net.to(device)
    net.load_state_dict(torch.load(weights_path))
    return net

def load_model2(weights_path: str = 'digit_mnist_model.pt', relative=True):
    if relative:
        weights_path = os.path.join(os.path.dirname(__file__), weights_path)
    net = Digit_OCR_CNN2()
    net.eval()
    net.to(device)
    net.load_state_dict(torch.load(weights_path))
    return net


def transform(img):
    out_img = img.T.copy(order='C')
    out_img = torch.tensor(out_img)
    out_img = out_img.unsqueeze(0)
    out_img = out_img.float()
    return out_img

def evaluate(net, img, verbose=False):
    tensor = transform(img).to(device)
    with torch.no_grad():
        output = net(tensor)
        _, predicted = torch.max(output.data, 1)
        pred_prob = F.softmax(output)[0][predicted].item() # Hypothetical probability

    if verbose:
        print(f'Ouput: {output}')
        print(f'Predicted: {predicted}')

    return predicted.item(), pred_prob

def multi_evaluate(net, img, verbose=False):
    
    guesses = np.zeros(10)
    raw_guesses = np.zeros(10)
    
    ax1sums = np.sum(img, axis=0)
    up_shift = np.argmax(ax1sums>0)
    down_shift = np.argmax(ax1sums[::-1]>0)
    
    ax2sums = np.sum(img, axis=1)
    left_shift = np.argmax(ax2sums>0)
    right_shift = np.argmax(ax2sums[::-1]>0)

    for x in range(-left_shift, right_shift+1):
        for y in range(-up_shift, down_shift+1):
            shifted_img = np.roll(img, (x, y), axis=(0,1))
            
            tensor = transform(shifted_img).to(device)
            with torch.no_grad():
                output = net(tensor)
                _, predicted = torch.max(output.data, 1)
                
                guesses[predicted] += 1
                raw_guesses += output.data.numpy().flatten()
                
    if verbose:       
        print(guesses)
        print(raw_guesses)
        
    return np.argmax(guesses)

def random_alignment(img):
    ax1sums = np.sum(img, axis=0)
    up_shift = np.argmax(ax1sums>0)
    down_shift = np.argmax(ax1sums[::-1]>0)
    
    ax2sums = np.sum(img, axis=1)
    left_shift = np.argmax(ax2sums>0)
    right_shift = np.argmax(ax2sums[::-1]>0)

    random_y = random.randint(-up_shift, down_shift)
    random_x = random.randint(-left_shift, right_shift)
    
    return np.roll(img, (random_x, random_y), axis=(0, 1))

def view(img):
    img = img.T.copy(order='C')

    new_img = np.repeat(img, 3)
    new_img = np.reshape(new_img, (28, 28, 3))

    new_img *= 255
    new_img = new_img.astype(np.uint8)
    cv2.imshow(f'img', new_img)
    cv2.waitKey(5)
