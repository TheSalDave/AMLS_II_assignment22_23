# Defining relevant constants
root_dir = 'Datasets/train_images' # path to image folder
label_file = 'Datasets/train.csv' # path to label file
IMG_SIZE = 256 # Image size as input to the model
BATCH_SIZE = 64 # Batch size used for training

# Importing libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.data.dataset import random_split
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from torch.utils.data.sampler import SubsetRandomSampler
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
from PIL import Image
from IPython.display import display
from torch.utils.data import Dataset
from PIL import Image
from sklearn.metrics import confusion_matrix

# Defining transforms / image augmentations
tensor_transform = transforms.Compose([
    transforms.ToTensor()
])

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Apply random horizontal flip with probability based on `do_mirror` parameter
    transforms.ColorJitter(contrast=0.2, hue=0.03, saturation=0.2),  # Apply color jitter with given ranges
    transforms.RandomAffine(degrees=20, shear=0.2),  # Apply random affine transformation with given ranges
    transforms.RandomPerspective(distortion_scale=0.1),  # Apply random perspective transformation with given range
    transforms.ToTensor()
])


class CustomDataset(Dataset):
    """A custom PyTorch dataset for loading, transforming and storing images with their corresponding labels.

        Args:
            root_dir (str): The root directory of the images.
            label_file (str): The path to the csv file containing the labels.
            transform (callable): Optional transform to be applied on a sample.

        Attributes:
            root_dir (str): The root directory of the images.
            label_file (str): The path to the csv file containing the labels.
            transform (callable): Optional transform to be applied on a sample.
            labels (list): The list of labels loaded from the csv file in ascending order based on hexadecimal file names of the corresponding images.
            image_files (list): The list of image files sorted in ascending order based on hexadecimal file names.

        Methods:
            __len__(): Returns the length of the dataset.
            __getitem__(idx): Returns the transformed image and its corresponding label for a given index.

        Returns:
            A PyTorch dataset instance.

        Example usage:
            dataset = CustomDataset(root_dir='./data', label_file='./data/labels.csv', transform=transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]))\n
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        """

    def __init__(self, root_dir, label_file, transform):
        self.root_dir = root_dir
        self.label_file = label_file
        self.transform = transform

        # Read the labels from the csv file
        self.labels = pd.read_csv(label_file, usecols=[1], skiprows = 1, names=['diagnosis'])['diagnosis'].tolist()

        # Get the list of image files in ascending order (according to hexadecimal image file name)
        self.image_files = []
        for root, dirs, files in os.walk(root_dir):
            # Sort the files based on hexadecimal image file names
            files = sorted(files, key=lambda x: int(x.split('.')[0], 16))
            self.image_files.extend([os.path.join(root, file) for file in files])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert('RGB')
        
        # Load the image and apply the transformation
        if self.transform:
            image = self.transform(image)
            # Apply pre processing steps (cropping and highlighting)
            image = preprocess(image)
        else: # else branch added to have access to the original images (for Original_images.png)
            image = transforms.ToTensor(image)

        # Convert the label to a tensor
        label = torch.tensor(self.labels[idx])
        
        return image, label
    

def crop_image_from_gray(image: np.ndarray, tolerance: int) -> np.ndarray:
    """
    Crops an image based on the grayscale intensity of the pixels in the image.
    
    Args:
        image (numpy.ndarray): The input image as a NumPy array.
        tolerance (int): The tolerance threshold to use for the grayscale mask.
        
    Returns:
        numpy.ndarray: The cropped image as a NumPy array.
    """

    image = image.numpy()
    image = np.transpose(image, (1, 2, 0))
    
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_img = np.round(gray_img * 255).astype(np.uint8)
    mask = gray_img > tolerance
    
    check_shape = image[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
    if (check_shape == 0): 
        print("Image is too dark. We would crop out everything.")
        return image # return original image
    else:
        image1=image[:,:,0][np.ix_(mask.any(1),mask.any(0))]
        image2=image[:,:,1][np.ix_(mask.any(1),mask.any(0))]
        image3=image[:,:,2][np.ix_(mask.any(1),mask.any(0))]
        
        image = np.stack([image1,image2,image3],axis=-1)

    return image


def highlighting(image: np.ndarray, sigmaX: float) -> np.ndarray:
    """
    Resizes the image and enhances the edges by applying a highlight effect.
    
    Args:
        image (numpy.ndarray): The input image as a NumPy array.
        sigmaX (float): The standard deviation of the Gaussian kernel to use for blurring the image.
        
    Returns:
        numpy.ndarray: The highlighted image as a NumPy array.
    """

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = np.round(image * 255).astype(np.uint8)
    image = cv2.addWeighted (image, 4, cv2.GaussianBlur(image, (0,0), sigmaX), -4, 128)
        
    return image


def preprocess(image: np.ndarray) -> torch.Tensor:
    """
    Applies all steps of the pre-processing pipeline to the image.
    \n\tStep 1:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Cropping uninformative areas
    \n\tStep 2:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Resizing the image
    \n\tStep 3:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Highlighting relevant regions
    \n\tStep 4:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Return image as pytroch tensor
    
    Args:
        image (numpy.ndarray): The input image as a NumPy array.

    Returns:
        torch.Tensor: The preprocessed image as a PyTorch tensor.
    """
    # Cropping the image
    processed_image = crop_image_from_gray(image,6)
    processed_image = highlighting(processed_image)
    processed_image = tensor_transform(processed_image)
    
    return processed_image


def main():
    """
    This method runs the complete, automated workflow that was developed for this assignment. 
    It generates relevant figures and saves them in the Figures/ folder. 
    As the final model was too large to be pushed to github the model needs to be retrained. 
    The training took approximately 10 hours when trained on Kaggle servers.
    """

    ##############################################################################################################
    #                                                                                                            #
    #                                      Part 1 - Pre-processing pipeline                                      #
    #                                                                                                            #
    ##############################################################################################################

    # Initializing the custom dataset
    original_data = CustomDataset(root_dir, label_file)
    no_aug_data = CustomDataset(root_dir, label_file, transform=tensor_transform)
    data = CustomDataset(root_dir, label_file, transform=train_transform)


    # Figure for original images
    plt.figure()
    fig, axes = plt.subplots(2, 5, figsize=(10, 6))

    # Loop through the first 10 elements of the dataset
    for i in range(10):
        # Access the image and label at index i
        image, label = original_data[i]
        
        # Convert the image to a PIL Image
        image = to_pil_image(image)
        
        # Get the row and column index for the subplot
        row_idx = i // 5
        col_idx = i % 5
        
        # Display the image with the label as the title on the corresponding subplot
        axes[row_idx][col_idx].imshow(image)
        axes[row_idx][col_idx].set_title(f"Diagnosis: {label.item()}")
        axes[row_idx][col_idx].axis('off')

    # Adjust the layout and save the figure
    plt.subplots_adjust(wspace=0.1, hspace=-0.3)
    if not os.path.exists('Figures/Original_images.png'):
        plt.savefig('Figures/Original_images.png', dpi=500)
        print('Saved Original_images.png')
    else:
        print('Original_images.png already exists')


    # Figure for no augmentation images
    plt.figure()
    fig, axes = plt.subplots(2, 5, figsize=(10, 6))

    # Loop through the first 10 elements of the dataset
    for i in range(10):
        # Access the image and label at index i
        image, label = no_aug_data[i]
        
        # Convert the image to a PIL Image
        image = to_pil_image(image)
        
        # Get the row and column index for the subplot
        row_idx = i // 5
        col_idx = i % 5
        
        # Display the image with the label as the title on the corresponding subplot
        axes[row_idx][col_idx].imshow(image)
        axes[row_idx][col_idx].set_title(f"Diagnosis: {label.item()}")
        axes[row_idx][col_idx].axis('off')

    # Adjust the layout and save the figure
    plt.subplots_adjust(wspace=0.1, hspace=-0.3)
    if not os.path.exists('Figures/No_Aug_images.png'):
        plt.savefig('Figures/No_Aug_images.png', dpi=500)
        print('Saved No_Aug_images.png')
    else:
        print('No_Aug_images.png already exists')


    # Figure for post pipeline images
    plt.figure()
    fig, axes = plt.subplots(2, 5, figsize=(10, 6))

    # Loop through the first 10 elements of the dataset
    for i in range(10):
        # Access the image and label at index i
        image, label = data[i]
        
        # Convert the image to a PIL Image
        image = to_pil_image(image)
        
        # Get the row and column index for the subplot
        row_idx = i // 5
        col_idx = i % 5
        
        # Display the image with the label as the title on the corresponding subplot
        axes[row_idx][col_idx].imshow(image)
        axes[row_idx][col_idx].set_title(f"Diagnosis: {label.item()}")
        axes[row_idx][col_idx].axis('off')

    # Adjust the layout and save the figure
    plt.subplots_adjust(wspace=0.1, hspace=-0.3)
    if not os.path.exists('Figures/Post_Pipeline_images.png'):
        plt.savefig('Figures/Post_Pipeline_images.png', dpi=500)
        print('Saved Post_Pipeline_images.png')
    else:
        print('Post_Pipeline_images.png already exists')


    ##############################################################################################################
    #                                                                                                            #
    #                                          Part 2 - Model training                                           #
    #                                                                                                            #
    ##############################################################################################################

    # Splitting the dataset into train and test (validation was only required for model selection and hyperparameter tuning)
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size

    # We use random seed 42 so that the unseen dataset is the same as it was during model selection and hyperparameter tuning.
    train_set, test_set = random_split(data, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    # Create the DataLoader objects for the training and test sets
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)
    print("Data split into train and test with corresponding dataloaders.")


    # Account for imbalanced dataset
    class_frequencies = torch.zeros(5)  # Calculate class frequencies
    total_samples = len(data)

    for label in data.labels:
        class_frequencies[label] += 1

    # Calculate class weights using inverse class frequency
    class_weights = total_samples / (class_frequencies * 5)
    print("Classweight losses calculated from inverse class frequencies:")
    print(class_weights)

    # Loading the pretrained model
    RegnetY16GF = models.regnet_y_16gf(weights=models.RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1)

    # Freezing parameters
    for param in RegnetY16GF.parameters():
        param.requires_grad = False
    
    # modifying classifier
    RegnetY16GF.fc = nn.Sequential(nn.Linear(3024,1000),nn.Linear(1000,5),nn.LogSoftmax(dim=1))

    # Defining loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight = class_weights)
    optimizer = torch.optim.Adam(RegnetY16GF.fc.parameters(), lr=0.001)

    # Unfreezing classification layers
    for param in RegnetY16GF.fc.parameters():
        param.requires_grad = True


    ############################################### Start training ###############################################
        
    start_time = time.time()

    epochs = 3

    # Limit nr of batches
    max_trn_batch = 34 
    max_test_batch = 11 

    train_losses = []
    test_losses = []
    train_correct = []
    test_correct = []

    for i in range(epochs):
        trn_corr = 0
        tst_corr = 0
        
        # Run the training batches
        for b, (X_train, y_train) in enumerate(train_loader):

            # Limit the number of batches
            if b == max_trn_batch:
                break

            b+=1 # For the calculations later

            # Apply the model
            # Convert input data to PyTorch tensor and specify the desired data type
            X_train = torch.tensor(X_train).to(torch.float32)  # Convert to float32
            # Transpose the dimensions of your input data to match the expected shape
            X_train = np.transpose(X_train, (0, 3, 1, 2))
            y_pred = RegnetY16GF(X_train)

            # calculate imbalance loss
            loss = criterion(y_pred, y_train)
    
            # Tally the number of correct predictions
            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr
            
            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print interim results
            if b%(17) == 0:
                print(f'epoch: {i:2}  |  batch: {b:4}  |  loss: {loss.item():10.8f}  |  accuracy: {trn_corr.item()*100/(64*b):7.2f}%   |   Duration: {time.time() - start_time:.0f} seconds')

        train_losses.append(loss.item())
        train_correct.append(trn_corr.item())

        # Run the testing batches
        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_loader):
                # Limit the number of batches for test
                if b == max_test_batch:
                    break

                # Apply the model
                X_test = torch.tensor(X_test).to(torch.float32)  # Convert to float32
                X_test = np.transpose(X_test, (0, 3, 1, 2))
                y_test_pred = RegnetY16GF(X_test)
                loss = criterion(y_test_pred, y_test)

                # Tally the number of correct predictions
                predicted = torch.max(y_test_pred.data, 1)[1] 
                tst_corr += (predicted == y_test).sum()

        test_losses.append(loss.item())
        test_correct.append(tst_corr.item())
    
    # Decrease learning rate
    optimizer = torch.optim.Adam(RegnetY16GF.fc.parameters(), lr=0.0001)

    # Continue training
    epochs = 3

    # Limit nr of batches
    max_trn_batch = 34 
    max_test_batch = 11

    for i in range(epochs):
        trn_corr = 0
        tst_corr = 0
        
        # Run the training batches
        for b, (X_train, y_train) in enumerate(train_loader):

            # Limit the number of batches
            if b == max_trn_batch:
                break

            b+=1 # For the calculations later

            # Apply the model
            # Convert input data to PyTorch tensor and specify the desired data type
            X_train = torch.tensor(X_train).to(torch.float32)  # Convert to float32
            # Transpose the dimensions of your input data to match the expected shape
            X_train = np.transpose(X_train, (0, 3, 1, 2))
            y_pred = RegnetY16GF(X_train)

            # calculate imbalance loss
            loss = criterion(y_pred, y_train)
    
            # Tally the number of correct predictions
            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr
            
            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print interim results
            if b%(17) == 0:
                print(f'epoch: {i:2}  |  batch: {b:4}  |  loss: {loss.item():10.8f}  |  accuracy: {trn_corr.item()*100/(64*b):7.2f}%   |   Duration: {time.time() - start_time:.0f} seconds')

        train_losses.append(loss.item())
        train_correct.append(trn_corr.item())

        # Run the testing batches
        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_loader):
                # Limit the number of batches for test
                if b == max_test_batch:
                    break

                # Apply the model
                X_test = torch.tensor(X_test).to(torch.float32)  # Convert to float32
                X_test = np.transpose(X_test, (0, 3, 1, 2))
                y_test_pred = RegnetY16GF(X_test)
                loss = criterion(y_test_pred, y_test)

                # Tally the number of correct predictions
                predicted = torch.max(y_test_pred.data, 1)[1] 
                tst_corr += (predicted == y_test).sum()

        test_losses.append(loss.item())
        test_correct.append(tst_corr.item())

    
    # Decrease learning rate
    optimizer = torch.optim.Adam(RegnetY16GF.fc.parameters(), lr=0.00001)

    # Continue training
    epochs = 4

    # Limit nr of batches
    max_trn_batch = 34 
    max_test_batch = 11

    for i in range(epochs):
        trn_corr = 0
        tst_corr = 0
        
        # Run the training batches
        for b, (X_train, y_train) in enumerate(train_loader):

            # Limit the number of batches
            if b == max_trn_batch:
                break

            b+=1 # For the calculations later

            # Apply the model
            # Convert input data to PyTorch tensor and specify the desired data type
            X_train = torch.tensor(X_train).to(torch.float32)  # Convert to float32
            # Transpose the dimensions of your input data to match the expected shape
            X_train = np.transpose(X_train, (0, 3, 1, 2))
            y_pred = RegnetY16GF(X_train)

            # calculate imbalance loss
            loss = criterion(y_pred, y_train)
    
            # Tally the number of correct predictions
            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr
            
            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print interim results
            if b%(17) == 0:
                print(f'epoch: {i:2}  |  batch: {b:4}  |  loss: {loss.item():10.8f}  |  accuracy: {trn_corr.item()*100/(64*b):7.2f}%   |   Duration: {time.time() - start_time:.0f} seconds')

        train_losses.append(loss.item())
        train_correct.append(trn_corr.item())

        # Run the testing batches
        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_loader):
                # Limit the number of batches for test
                if b == max_test_batch:
                    break

                # Apply the model
                X_test = torch.tensor(X_test).to(torch.float32)  # Convert to float32
                X_test = np.transpose(X_test, (0, 3, 1, 2))
                y_test_pred = RegnetY16GF(X_test)
                loss = criterion(y_test_pred, y_test)

                # Tally the number of correct predictions
                predicted = torch.max(y_test_pred.data, 1)[1] 
                tst_corr += (predicted == y_test).sum()

        test_losses.append(loss.item())
        test_correct.append(tst_corr.item())

        print("Training finished. Starting evaluation")

    ##############################################################################################################
    #                                                                                                            #
    #                                          Part 3 - Evaluation                                               #
    #                                                                                                            #
    ##############################################################################################################
    
    # Loss learning curve
    print("Train losses:")
    print(train_losses)
    print("Test losses:")
    print(test_losses)

    plt.figure()
    plt.plot(train_losses, label='training loss')
    plt.plot(test_losses, label='test loss')
    plt.title('Loss at the end of each epoch')
    plt.legend()

    # Save the figure
    if not os.path.exists('Figures/Loss_learning_curve.png'):
        plt.savefig('Figures/Loss_learning_curve.png', dpi=500)
        print('Saved Loss_learning_curve.png')
    else:
        print('Loss_learning_curve.png already exists')


    # Accuracy learning curve
    print(f'Number of correct prediction on train set: {train_correct}')
    print(f'Number of correct prediction on test set: {test_correct}')
    print(f'Test accuracy: {test_correct[-1]*100/704:.3f}%')
    plt.plot([t/21.76 for t in train_correct], label='train accuracy')
    plt.plot([t/7.04 for t in test_correct], label='test accuracy')
    plt.title('Accuracy at the end of each epoch')
    plt.legend();

    # Saving the figure
    if not os.path.exists('Figures/Accuracy_learning_curve.png'):
        plt.savefig('Figures/Accuracy_learning_curve.png', dpi=500)
        print('Saved Accuracy_learning_curve.png')
    else:
        print('Accuracy_learning_curve.png already exists')


    # Rerunning final model on the test set
    actual = []
    predictions = []

    # Run the testing batches
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            actual.extend(y_test.tolist())

            # Apply the model
            # Convert input data to PyTorch tensor and specify the desired data type
            X_test = torch.tensor(X_test).to(torch.float32)  # Convert to float32
            # Transpose the dimensions of your input data to match the expected shape
            X_test = np.transpose(X_test, (0, 3, 1, 2))
            y_val = RegnetY16GF(X_test)

            # Tally the number of correct predictions
            predicted = torch.max(y_val.data, 1)[1]
            predictions.extend(predicted.tolist())

    act = np.reshape(np.array(actual), (test_size,))
    preds = np.reshape(np.array(predictions), (test_size,))

    # Confusion matrix
    plt.figure()
    conf_matrix = confusion_matrix(y_true=act, y_pred=preds)

    # Print the confusion matrix using Matplotlib
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actual diagnosis', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    
    # Saving the figure
    if not os.path.exists('Figures/Confusion_matrix.png'):
        plt.savefig('Figures/Confusion_matrix.png', dpi=500)
        print('Saved Confusion_matrix.png')
    else:
        print('Confusion_matrix.png already exists')


if __name__ == "__main__":
    main()