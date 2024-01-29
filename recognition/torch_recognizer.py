import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2

from time import sleep # DEBUG

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Digit_OCR_CNN(nn.Module):
    # Simple network, nn.CrossEntropyLoss optimizer
    def __init__(self):
        super(Digit_OCR_CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 6, 3, padding=1)
        
        self.drop = nn.Dropout(0.2)

        self.fc1 = nn.Linear(6 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, state):
        x = self.pool(F.relu(self.conv1(state)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop(x)
        x = x.view(-1, 6 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)
        return x

class Digit_OCR_CNN2(nn.Module):
    # Larger network, softmax output
    def __init__(self):
        super(Digit_OCR_CNN2, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 6, 3, padding=1)
        
        self.drop = nn.Dropout(0.2)

        self.fc1 = nn.Linear(6 * 14 * 14, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        
        self.softact = nn.Softmax()
        
    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop(x)
        x = x.view(-1, 6 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.softact(self.fc3(x))

        return x
    
class Digit_OCR_CNN3(nn.Module):
    # Agressive dropout
    def __init__(self):
        super(Digit_OCR_CNN3, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 6, 3, padding=1)

        self.fc1 = nn.Linear(6 * 14 * 14, 256)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 256)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 10)
        self.softact = nn.Softmax()
        
    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 6 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.softact(self.fc3(x))

        return x

class Digit_OCR_CNN4(nn.Module):
    # Log softmax, nn.NLLLoss optiizer
    def __init__(self):
        super(Digit_OCR_CNN4, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)
        self.conv2 = nn.Conv2d(3, 6, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(6 * 14 * 14, 128)
        self.drop2 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 128)
        self.drop3 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, state):
        # print(state.size())

        x = F.relu(self.conv1(state))
        # x = self.pool(F.relu(self.conv1(state)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop1(x)
        # print(x.size())

        x = x.view(-1, 6 * 14 * 14)
        # print(x.size())

        x = F.relu(self.fc1(x))
        x = self.drop2(x)

        x = F.relu(self.fc2(x))
        x = self.drop3(x)
        x = F.log_softmax(self.fc3(x), dim=1)

        return x


class Digit_OCR_CNN5(nn.Module):

    def __init__(self):
        super(Digit_OCR_CNN5, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)
        self.conv2 = nn.Conv2d(3, 6, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(6 * 14 * 14, 128)
        self.drop2 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 128)
        self.drop3 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, state):
        x = F.relu(self.conv1(state))
        # x = self.pool(F.relu(self.conv1(state)))
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.size())
        x = self.drop1(x)
        x = x.view(-1, 6 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)

        x = F.relu(self.fc2(x))
        x = self.drop3(x)
        x = F.log_softmax(self.fc3(x), dim=1)

        return x

def load_model(weights_path: str = 'digit_mnist_model.pt'):
    net = Digit_OCR_CNN()
    net.eval()
    net.to(device)
    net.load_state_dict(torch.load(weights_path))
    return net

def load_model2(weights_path: str = 'digit_mnist_model2.pt'):
    net2 = Digit_OCR_CNN2()
    net2.eval()
    net2.to(device)
    net2.load_state_dict(torch.load(weights_path))
    return net2

def load_model3(weights_path: str = 'digit_mnist_model3.pt'):
    net3 = Digit_OCR_CNN3()
    net3.eval()
    net3.to(device)
    net3.load_state_dict(torch.load(weights_path))
    return net3

def load_model4(weights_path: str = 'digit_mnist_model4.pt'):
    net4 = Digit_OCR_CNN4()
    net4.eval()
    net4.to(device)
    net4.load_state_dict(torch.load(weights_path))
    return net4

def load_model5(weights_path: str = 'digit_mnist_model5.pt'):
    net5 = Digit_OCR_CNN4()
    net5.eval()
    net5.to(device)
    net5.load_state_dict(torch.load(weights_path))
    return net5

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
    if verbose:
        print(f'Ouput: {output}')
        print(f'Predicted: {predicted}')

    return predicted.item()

def view(img):
    img = img.T.copy(order='C')

    new_img = np.repeat(img, 3)
    new_img = np.reshape(new_img, (28, 28, 3))

    new_img *= 255
    new_img = new_img.astype(np.uint8)
    cv2.imshow(f'img', new_img)
    cv2.waitKey(5)
