# Animal Classification

We have developed a neural network for classifying 10 different types of animals. The training dataset we used is sourced from Kaggle.com. To contribute testing, we have made a small training dataset available in our repository. Here are the translations for the animal classes(Italian - English):

1. "cane": "dog"
2. "cavallo": "horse"
3. "elefante": "elephant"
4. "farfalla": "butterfly"
5. "gallina": "chicken"
6. "gatto": "cat"
7. "mucca": "cow"
8. "pecora": "sheep"
9. "scoiattolo": "squirrel"
10. "ragno": "spider"

You can access the dataset we used for training at the following link: [Dataset Link](https://drive.google.com/drive/folders/1Wm8gLNpSvOoiDtiqUVG4-JIRfcmrrvCl?usp=sharing). There you also can find our networks.

PyTorch - **ResNet50** trained on IMAGENET1K_V2. You need download `resnet50_torch.pth` (Only this network src already include.)

TensorFlow - **GoogleNet/IncpetionV3** trained on ImageNet. You need download `Tuned_Inseption.rar`.

We have implemented two separate networks for this task, one using TensorFlow and the other using PyTorch. You have the flexibility to choose either of these frameworks for the implementation.
