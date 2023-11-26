# Flask_WebApp_MissingBottles
A web application for detection missing bottles by counting the existing ones. 


# Automated Detection of Missing Bottles

## Overview

This project uses a pre-trained YOLOv8 model to automatically detect water bottles within packages. It leverages computer vision to inspect packages, ensuring the correct number of water bottles is present. This system can be integrated into logistics processes to prevent dispatching packages with missing items.

## Usage

### Training the Model

Prepare a dataset containing annotated images of packages with water bottles. Train the YOLOv8 model using the provided dataset.

### Inference

Perform inference on saved-images or a live video using the trained model. Load the model and provide the path to the image for inference. The system will identify and display any missing bottles in the image.

### Integration

Integrate the detection system into your logistics workflow. Ensure it can process live video feeds or images, and provide alerts for packages with missing bottles.

## Contributing

Contributions are welcome! If you have suggestions, bug fixes, or improvements, feel free to open an issue or create a pull request.

