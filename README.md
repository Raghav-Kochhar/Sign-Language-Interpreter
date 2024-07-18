# SignLanguageInterpreter

SignLanguageInterpreter is an advanced AI model that predicts sign language from images and videos. It can interpret sign language gestures and translate them into text, as well as convert text into sign language representations.

# CAUTION: YOU WILL HAVE TO DOWNLOAD A HUGE DATASET AND TRAIN THE MODEL ON IT FOR A LONG TIME TO GET GOOD RESULTS

## Installation

1. Clone the repository:

   `gh repo clone Raghav-Kochhar/Sign-Language-Interpreter`
   
2. Install the required dependencies:
   
   pip: `pip install -r pip_env.txt`

   conda: `conda env create -f conda_env.yml`

3. Download the model I have already trained and save it in the models folder: [https://drive.google.com/file/d/1--SnAVYGYv3SLtUVYm3Z5UkFV-Z1725D/view?usp=drive_link](https://drive.google.com/file/d/13OKqHz0YFJo73eT7jn-6HAKW2VHAhiYp/view?usp=drive_link)

4. Set up Google Drive integration for data storage (follow Google Colab instructions if using Colab)

## Usage

1. Prepare your config.yaml file with necessary parameters

2. Run the Dataset downloader script:

   `python script.py`

3. Run the main script:

   `python app.py`
