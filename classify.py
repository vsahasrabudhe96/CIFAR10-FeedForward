import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import sys
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

#converting it to a tensor and normalizing it with mean an standard deviation
transform = transforms.Compose([transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = torchvision.datasets.CIFAR10(
    root='./model/data.cifar10', # location of the dataset
    train=True,                                     # this is training data
    transform=transform,    # Converts a PIL.Image or numpy.ndarray to torch.FloatTensor of shape (C x H x W)
    download=True                                   # if you haven't had the dataset, this will automatically download it for you
)

train_loader = Data.DataLoader(dataset=train_data, batch_size=6000,shuffle = True ) #setting the batch size to be 6000, and setting the shuffle parameter as true ensures that the iimages in the batches will be randomly shuffled

test_data = torchvision.datasets.CIFAR10(root='./model/data.cifar10/', train=False, transform=transform)

test_loader = Data.DataLoader(dataset=test_data, batch_size=5000,shuffle = True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') #the 10 cimage classes are stored in the tuple
epochs = 10 #number of epochs for each the model will run

class CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        self.hl1 = nn.Linear(in_features=32*32*3,out_features=1024) #Linear layer 1 with in-3076, out-1024
        self.relu1  =nn.ReLU()                                      #adding a ReLU layer for non linearity
        self.hl2 = nn.Linear(in_features = 1024,out_features=600)   #Linear layer 2 with in-1024, out-600
        self.relu2 = nn.ReLU()
        self.outp = nn.Linear(in_features=600,out_features = 10)    #Linear layer 3 with in- 600, out-10

    def forward(self,inp):
        inp = inp.view(-1,32*32*3) #reshaping the input to (1x3076) array
        inp = F.relu(self.hl1(inp)) #passing the input through the first linear layer and ReLU layer
        inp = F.relu(self.hl2(inp))
        inp = self.outp(inp)
        return inp

model = CIFAR10()

optimizer = torch.optim.Adam(model.parameters(),lr = 0.0005) #using Adam optimizer and setting the learning rate as 0.0005(we can also use SGD and rmsprop)
loss_func = nn.CrossEntropyLoss() #We make us cross entropy loss as the loss function(we can also make use of NLLLoss and log softmax)

#########TRAINING FUNCTION###########
def train():
     total_loss = 0
     total_train = 0
     correct_train = 0
     for images,labels in train_loader:
        model.train() #setiing the model in training mode
        optimizer.zero_grad() #every time train we want to gradients to be set to zero 
        output = model(images) #making the forward pass through the model
        loss = loss_func(output,labels) 
        loss.backward()
        optimizer.step()
    
        total_loss = total_loss + loss.item()
    
        #accuracy
        _, predicted = torch.max(output.data, 1) #we check the label which has maximum probability  
        total_train += labels.size(0)
        correct_train += predicted.eq(labels.data).sum().item()
        train_accuracy = 100 * correct_train / total_train
     return (train_accuracy,loss.item())

def test(epochs):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        with torch.no_grad(): 
            for data in test_loader:
                images, labels = data
                outputs = model(images) #forward pass
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0) #to get the total number of labels
                correct += (predicted == labels).sum().item() #we check the amount of predicted and actual labels which are same and then sum them for calculating the accuracy
                acc = 100*correct/total
    return acc

def predict(img_path):
    
    #Loading the model
    model = torch.load('./model/model.pt') #loading the model from the model path
    
    #Loadind the test image
    from PIL import Image
    img = Image.open(img_path)
    img_re = img.resize((32,32))
    
    with torch.no_grad():
        trans1 = transforms.ToTensor()
        img_tensor = trans1(img_re) #shape = [3, 32, 32]
        
        #Image Transformation
        trans = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        img_tensor = trans(img_tensor)
        
        single_image_batch = img_tensor.unsqueeze(0) #shape = [1, 3, 32, 32]
        outputs = model(single_image_batch)
        _, predicted = torch.max(outputs.data, 1)
        class_id = predicted[0].item()
        predicted_class = classes[predicted[0].item()]
        print("Predicted Class : {}".format(predicted_class))

    plt.rcParams["figure.figsize"] = (2, 2)
    plt.title(predicted_class)
    plt.imshow(img)
    plt.show()

def main():
    print("Training for {} epochs".format(epochs))
    for epoch in range(epochs):
        (train_model,loss_tr)=train()
        test_model = test(epochs)
        print('Epoch {}, train Loss: {:.3f}'.format(epoch ,loss_tr), "Training Accuracy {}:" .format(train_model), 'Test Accuracy {}'.format(test_model))
    print("Model successfully saved at ./model/")
    torch.save(model, './model/model.pt') #saving the model to the mentioned model path


if __name__ == '__main__':
    if len(sys.argv)==1:
        print("Please enter more commands for training and testing e.g 'train','test'")
        sys.exit()

    elif len(sys.argv)==2:
        try:
            assert sys.argv[1] == "train"
            main()
            sys.exit()
        except AssertionError:
            print('Please enter a valid command')
            sys.exit()

    elif len(sys.argv)==3:
        try:
            assert sys.argv[1] == "test"
            if not sys.argv[2].split('.')[-1].endswith('png'):
                print('Only PNG files supported')
                sys.exit()
            else:
                predict(sys.argv[2])
        except AssertionError:
            print('Please enter a valid command')
            sys.exit()
