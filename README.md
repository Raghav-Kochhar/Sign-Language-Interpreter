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
   
2. Install the required dependencies:
   
   for pip users: `pip install -r pip_env.txt`

   for conda users: `conda env create -f conda_env.yml`

3. Download the model I have already trained and save it in the models folder: https://drive.google.com/file/d/1--SnAVYGYv3SLtUVYm3Z5UkFV-Z1725D/view?usp=drive_link

4. Set up Google Drive integration for data storage (follow Google Colab instructions if using Colab)

## Usage

1. Prepare your config.yaml file with necessary parameters

2. Run the Dataset downloader script:

   `python script.py`

3. Run the main script:

   `python app.py`
