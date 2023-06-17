import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from PIL import Image


class AnimalClassifier:
    """
    AnimalClassifier is a class that uses a pre-trained ResNet-50 model to classify images of animals.
    """

    def __init__(self):
        """
        Initializes an instance of the AnimalClassifier class.
        """
        self.weights_path = "./resnet50_torch.pth"  # Path to the pre-trained weights file

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # Set the device to CUDA if available, otherwise use CPU
        self.labels = {
            0: "dog",
            1: "horse",
            2: "elephant",
            3: "butterfly",
            4: "rooster",
            5: "cat",
            6: "cow",
            7: "sheep",
            8: "spider",
            9: "squirrel"
        }  # Dictionary mapping class indices to animal labels

        self.model = resnet50(weights="IMAGENET1K_V2")  # Load the ResNet-50 model pre-trained on the ImageNet dataset
        weights = ResNet50_Weights.DEFAULT
        self.transform = weights.transforms()  # Get the transformation pipeline for preprocessing images

        self.model.fc = nn.Sequential(
            nn.Linear(2048, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )  # Replace the fully connected layer of the model with a new sequential layer

        self.model.load_state_dict(torch.load(self.weights_path))  # Load the pre-trained weights into the model

        self.model.to(self.device)  # Move the model to the specified device (CPU or GPU)

        self.model.eval()  # Set the model to evaluation mode

    def __call__(self, path):
        """
        Classifies an animal image given its file path.

        Args:
            path (str): The file path to the animal image.

        Returns:
            str: The predicted animal label.
        """
        img = Image.open(path)  # Open the image file
        image = self.transform(img)  # Apply the transformation pipeline to preprocess the image
        image = image.unsqueeze(0)  # Add a batch dimension to the image tensor

        image = image.to(self.device)  # Move the preprocessed image tensor to the specified device

        output = self.model(image)  # Forward pass through the model to get the output logits

        _, pred = torch.max(output, 1)  # Get the predicted class index

        pred = pred.to(torch.device("cpu"))  # Move the predicted class index to the CPU

        pred = self.labels[pred.item()]  # Get the corresponding animal label from the dictionary

        return pred
