# People-Clothing-Segmentation
# People Clothing Segmentation

This project implements a deep learning model to perform semantic segmentation of people and their clothing into multiple classes such as accessories, bags, shoes, glasses, hats, and more. The model uses a U-Net architecture with a pre-trained **MobileNetV2** as the backbone for feature extraction, enabling efficient and accurate segmentation on relatively small datasets.

## Features

- **U-Net Architecture:** The U-Net model is designed for image segmentation tasks, where it captures features from different resolutions using skip connections.
- **MobileNetV2 Backbone:** A pre-trained MobileNetV2 model is used for the downsampling part of the U-Net, making the model lightweight and efficient while maintaining high accuracy.
- **Multi-class Segmentation:** The model can identify and segment multiple classes (e.g., clothes, shoes, accessories) from a given image.
- **Visualization:** The training process includes visualizations of both ground truth masks and predicted masks, allowing easy monitoring of performance.
- **Accuracy Tracking:** The training script automatically tracks the accuracy and loss of the model on both training and validation datasets, providing detailed logs and plots.

## Requirements

To run the code, make sure you have the following Python libraries installed:

```bash
pip install tensorflow
pip install opencv-python
pip install matplotlib
pip install tqdm
pip install scikit-learn
