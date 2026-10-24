
import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path

# These two lines can be used in case you have multiple versions of the OpenMP library 
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' 


########################################################################
# How do we run these neural networks on the GPU?
#
# Training on GPU
# ----------------
# Just like how you transfer a Tensor onto the GPU, you transfer the neural
# net onto the GPU.
#
# Let's first define our device as the first visible cuda device if we have
# CUDA available:

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

########################################################################
# .. note::
#     If running on Windows and you get a BrokenPipeError, try setting
#     the num_worker of torch.utils.data.DataLoader() to 0.

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4     #You can change the batch size if you receive an out of memory error

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

########################################################################
# Let us show some of the training images, for fun.

import matplotlib.pyplot as plt
import numpy as np

# functions to unnormalize and show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

 
# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)
 
# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


########################################################################
# 2. Define a Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.

#Updating the Traning epoch to 20
n_epochs=20
average_epoch_loss={epoch+1:0 for epoch in range(n_epochs)}
for epoch in range(n_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    epoch_losses = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        epoch_losses += loss.item()
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

    average_epoch_loss[epoch+1] = epoch_losses/len(trainloader)
    print(f'[{epoch + 1}] average loss: {average_epoch_loss[epoch+1]:.3f}')

print('Finished Training')

plt.plot(average_epoch_loss.keys(),average_epoch_loss.values(),marker='o')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid(True)


plot_save = Path(__file__).resolve().parent / 'problem1_average_loss.png'
plt.savefig(plot_save)
plt.show()

########################################################################
# Let's quickly save our trained model:
# Saving it in the current Problem Folder
PATH = Path(__file__).resolve().parent / './cifar_net.pth'
torch.save(net.state_dict(), PATH)

########################################################################
# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We have trained the network for 2 passes over the training dataset.
# But we need to check if the network has learnt anything at all.
#
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
#
# Okay, first step. Let us display an image from the test set to get familiar.

dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

########################################################################
# Next, let's load back in our saved model (note: saving and re-loading the model
# wasn't necessary here, we only did it to illustrate how to do so):

net = Net()
net.load_state_dict(torch.load(PATH))

########################################################################
# Okay, now let us see what the neural network thinks these examples above are:

outputs = net(images)

########################################################################
# The outputs are energies for the 10 classes.
# The higher the energy for a class, the more the network
# thinks that the image is of the particular class.
# So, let's get the index of the highest energy:
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))

########################################################################
# The results seem pretty good.
#
# Let us look at how the network performs on the whole dataset.

# Accuracy on Training Dataset

correct_training=0
total_training=0

# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in trainloader:
        images,labels=data
        # calculate outputs by running images through the network
        outputs=net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted=torch.max(outputs.data,1)
        total_training+=labels.size(0)
        correct_training += (predicted==labels).sum().item()

print(f'Accuracy of the network on the training images: {100 * correct_training / total_training:.2f} %')


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f} %')

########################################################################
# That looks way better than chance, which is 10% accuracy (randomly picking
# a class out of 10 classes).
# Seems like the network has learned something.
#
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:

#Accuracy for Training dataset
training_dataset_correct_pred = {classname:0 for classname in classes}
training_dataset_total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in trainloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                training_dataset_correct_pred[classes[label]] += 1
            training_dataset_total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in training_dataset_correct_pred.items():
    accuracy = 100 * float(correct_count) / training_dataset_total_pred[classname]
    print(f'Accuracy for class in Training Dataset: {classname:5s} is {accuracy:.1f} %')

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


## Grad cam Implementation, capture heatmap for a prediction to show the reason for model selection

# One random image
image, label = testset[0]
x=image.unsqueeze(0)

net.eval()
net.zero_grad(set_to_none=True)

# Saving intermediate features by running model manually
conv1_layer_features=F.relu(net.conv1(x))
conv1_layer_features.retain_grad()
x = net.pool(conv1_layer_features)


conv2_layer_features=F.relu(net.conv2(x))
conv2_layer_features.retain_grad()
x=net.pool(conv2_layer_features)


x = torch.flatten(x,1)
x = F.relu(net.fc1(x))
x = F.relu(net.fc2(x))
prediction_scores=net.fc3(x)

result_class=prediction_scores.argmax(dim=1).item()
prediction_scores[0,result_class].backward()

true_class = classes[label]
predicted_class = classes[result_class]

print(f'True class: {true_class}')
print(f'Model prediction: {predicted_class}')

# Starting the grad cam calculations according to the paper 
conv1_weights = conv1_layer_features.grad.mean(dim=(2,3), keepdim=True)
grad_cam_conv1=F.relu((conv1_weights*conv1_layer_features).sum(dim=1, keepdim=True))

conv2_weights = conv2_layer_features.grad.mean(dim=(2,3), keepdim=True)
grad_cam_conv2=F.relu((conv2_weights*conv2_layer_features).sum(dim=1, keepdim=True))

# Resizing the Matrix to match input

grad_cam_conv1=F.interpolate(grad_cam_conv1, size=(32,32), mode='bilinear',align_corners=False)[0,0].detach()
grad_cam_conv2=F.interpolate(grad_cam_conv2, size=(32,32), mode='bilinear',align_corners=False)[0,0].detach()

# Display the result

grad_cam_conv1=grad_cam_conv1/(grad_cam_conv1.max() + 1e-8)
grad_cam_conv2=grad_cam_conv2/(grad_cam_conv2.max() + 1e-8)

image_unnormal=image.permute(1,2,0).numpy()/2+0.5

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.imshow(image_unnormal)
plt.title('Input Image')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(image_unnormal)
plt.imshow(grad_cam_conv1.numpy(), cmap='coolwarm', alpha=0.55)
plt.title('Convolution 1 layer Heatmap')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(image_unnormal)
plt.imshow(grad_cam_conv2.numpy(), cmap='coolwarm', alpha=0.55)
plt.title('Convolution 2 layer Heatmap')
plt.axis('off')
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / 'problem1_gradcam_layers.png')
plt.show()