import streamlit as st
import cv2
import torch
import numpy as np
from torchvision import transforms
from transformers import BertTokenizer, BertForSequenceClassification
from data_preparation import MSASLDataset

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "data/MS-ASL"
dataset = MSASLDataset(data_dir, "train")
num_classes = len(dataset.classes)

# Load the TorchScript model
sign_model = torch.jit.load("models/sign_language_model_torchscript.pt")
sign_model.to(device)
sign_model.eval()

# Load BERT model
bert_model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=num_classes
)
bert_model.to(device)
bert_model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def predict_sign(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = transform(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        output = sign_model(frame)
        _, predicted = torch.max(output, 1)

    class_name = dataset.get_class_name(predicted.item())
    synonyms = dataset.get_synonyms(class_name)
    return class_name, synonyms


st.title("Sign Language Detection")

run = st.checkbox("Run")
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

# Create placeholders for sign and synonyms
sign_placeholder = st.empty()
synonyms_placeholder = st.empty()

while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

    # Predict the sign
    sign, sign_synonyms = predict_sign(frame)

    # Update the placeholders with new content
    sign_placeholder.write(f"Detected Sign: {sign}")
    synonyms_placeholder.write(f"Sign Synonyms: {', '.join(sign_synonyms)}")

    # Add a small delay to reduce CPU usage
    cv2.waitKey(100)

st.write("Stopped")
