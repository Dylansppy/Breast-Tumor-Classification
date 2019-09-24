import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


from preprocessing import list_pictures
import cv2
import os, time
import pandas as pd
from keras.preprocessing import image
from keras.models import Model
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input

# Extract features with VGG16, the default input size for this model is 224x224.
img_path = os.path.join('..', 'data', 'breast_processed')
features2 = []
model2 = VGG16(weights='imagenet', include_top=False)

def feature_extraction(model, ima_path)
    for img_dir in list_pictures(img_path):
        img = cv2.imread(img_dir)
        img = cv2.resize(img, (224, 224))
        #x = image.img_to_array(img)
        #x_list.append(img)
        x = np.expand_dims(img, axis=0)
        x = preprocess_input(x)
        directory, file_name = os.path.split(img_dir)
        label = file_name.split('_')[1]
        if label == 'M':
            y = 1
        else:
            y = 0
        #x = np.expand_dims(x, axis=0)
        #x = preprocess_input(x)
        #x_df = pd.DataFrame(x_list)
        #y_df = pd.DataFrame(y_list)
        feature = model.predict(x)
        features2.append(feature)
    pd.DataFrame(features2)

# Extract features with Xception, the default input size for this model is 299x299.
img_path = os.path.join('..', 'data', 'breast_processed')
features = []
model1 = Xception(weights='imagenet', include_top=False)
for img_dir in list_pictures(img_path):
    img = cv2.imread(img_dir)
    img = cv2.resize(img, (299, 299))
    #x = image.img_to_array(img)
    #x_list.append(img)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    directory, file_name = os.path.split(img_dir)
    label = file_name.split('_')[1]
    if label == 'M':
        y = 1
    else:
        y = 0
    y_list.append(y)
    #x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)
    #x_df = pd.DataFrame(x_list)
    #y_df = pd.DataFrame(y_list)
    feature = model1.predict(x)
    features.append(feature)
pd.DataFrame(features)

# Extract features from an arbitrary intermediate layer with VGG19, the default input size for this model is 224x224
img_path = os.path.join('..', 'data', 'breast_processed')
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

base_model = VGG19(weights='imagenet')
model3 = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
block4_pool_features = model3.predict(x)

#Function to train the model
def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid accuracy: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Train a model with a pre-trained network
num_epochs = 10
if use_gpu:
    print ("Using GPU: "+ str(use_gpu))
    model = model.cuda()

# NLLLoss because our output is LogSoftmax
criterion = nn.NLLLoss()

# Adam optimizer with a learning rate
#optimizer = optim.Adam(model.fc.parameters(), lr=0.005)
optimizer = optim.SGD(model.fc.parameters(), lr = .0006, momentum=0.9)
# Decay LR by a factor of 0.1 every 5 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=10)


# Do validation on the test set
def test(model, dataloaders, device):
    model.eval()
    accuracy = 0

    model.to(device)

    for images, labels in dataloaders['valid']:
        images = Variable(images)
        labels = Variable(labels)
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        ps = torch.exp(output)
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()

        print("Testing Accuracy: {:.3f}".format(accuracy / len(dataloaders['valid'])))

test(model, dataloaders, device)


# Save the checkpoint
model.class_to_idx = dataloaders['train'].dataset.class_to_idx
model.epochs = num_epochs
checkpoint = {'input_size': [3, 224, 224],
                 'batch_size': dataloaders['train'].batch_size,
                  'output_size': 2,
                  'state_dict': model.state_dict(),
                  'data_transforms': data_transforms,
                  'optimizer_dict':optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'epoch': model.epochs}
torch.save(checkpoint, '8960_checkpoint.pth')