# SignLanguageInterpreter

## Project Overview
SignLanguageInterpreter is an advanced AI model that predicts sign language from images and videos. It can interpret sign language gestures and translate them into text, as well as convert text into sign language representations.

## Features
- Predict sign language from uploaded images
- Interpret sign language videos and convert them to text
- Convert text phrases to sign language representations
- Bidirectional translation between sign language and text

## Installation

1. Clone the repository:

   `gh repo clone Raghav-Kochhar/Sign-Language-Interpreter`

   `cd Sign-Language-Interpreter`
3. Install the required dependencies:

   `pip install -r requirements.txt`
5. Set up Google Drive integration for data storage (follow Google Colab instructions if using Colab)

## Usage

1. Prepare your config.yaml file with necessary parameters
2. Run the Dataset downloader script:
   `python script.py`
4. Run the main script:
   `python app.py`

## Data Preparation
The project uses the MSASL (Microsoft American Sign Language) dataset. The script handles:
- Downloading videos from various sources
- Processing and trimming videos
- Organizing data for training, validation, and testing

## Model Architecture
- Based on EfficientNetV2-S
- Fine-tuned for sign language recognition
- Uses transfer learning with frozen early layers

## Training
- Utilizes PyTorch for model training
- Implements data augmentation techniques
- Uses mixed precision training for efficiency
- Includes early stopping and learning rate scheduling

## Evaluation
The model is evaluated on a separate test set, with accuracy metrics provided.

## Model Export
The trained model is exported in both PyTorch and TorchScript formats for deployment.
