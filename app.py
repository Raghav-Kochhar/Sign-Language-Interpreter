import streamlit as st
import cv2
import torch
import numpy as np
from torchvision import transforms
from script import MSASLDataset, load_config, SignLanguageModel
from typing import List
import tempfile
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

config = load_config("config.yaml")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    dataset = MSASLDataset(
        config["data_dir"], config["processed_video_folder"], "train"
    )
    num_classes = len(dataset.classes)
except Exception as e:
    st.error(f"Error loading dataset: {str(e)}")
    st.stop()

try:
    model = SignLanguageModel(
        num_classes=num_classes, sequence_length=config["sequence_length"]
    )
    model.load_state_dict(torch.load(config["model_save_path"], map_location=device))
    model.to(device)
    model.eval()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@st.cache_data
def predict_sequence(frames: List[np.ndarray]) -> str:
    processed_frames = torch.stack([transform(frame).to(device) for frame in frames])
    processed_frames = processed_frames.unsqueeze(0)

    with torch.no_grad():
        output = model(processed_frames)
        _, predicted = torch.max(output, 1)

    predicted_signs = [dataset.get_class_name(idx.item()) for idx in predicted]
    return " ".join(predicted_signs)


@st.cache_data
def get_synonyms(sign):
    return dataset.get_synonyms(sign)


st.title("Sign Language Interpreter")

input_type = st.radio("Choose input type:", ("Webcam", "Video Upload"))

if input_type == "Webcam":
    run = st.checkbox("Start Interpreting")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    interpretation_placeholder = st.empty()
    synonyms_placeholder = st.empty()

    frames_buffer = []
    frame_skip = 2
    frame_count = 0

    while run:
        _, frame = camera.read()
        frame_count += 1

        if frame_count % frame_skip == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

            frames_buffer.append(frame)
            if len(frames_buffer) == config["sequence_length"]:
                sign_sequence = predict_sequence(frames_buffer)
                interpretation_placeholder.write(
                    f"Interpreted Sequence: {sign_sequence}"
                )

                synonyms = [get_synonyms(sign) for sign in sign_sequence.split()]
                synonyms_text = ", ".join(
                    [
                        f"{sign}: {', '.join(syns)}"
                        for sign, syns in zip(sign_sequence.split(), synonyms)
                    ]
                )
                synonyms_placeholder.write(f"Synonyms: {synonyms_text}")

                frames_buffer.pop(0)

        cv2.waitKey(100)

    camera.release()

else:
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            video = cv2.VideoCapture(tmp_file_path)
            frames_buffer = []
            all_predictions = []

            progress_bar = st.progress(0)

            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            for i in range(frame_count):
                ret, frame = video.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_buffer.append(frame)

                if len(frames_buffer) == config["sequence_length"]:
                    sign_sequence = predict_sequence(frames_buffer)
                    all_predictions.append(sign_sequence)
                    frames_buffer.pop(0)

                progress_bar.progress((i + 1) / frame_count)

            video.release()

            st.write("Interpretation Results:")
            for idx, prediction in enumerate(all_predictions):
                st.write(f"Sequence {idx + 1}: {prediction}")

                synonyms = [get_synonyms(sign) for sign in prediction.split()]
                synonyms_text = ", ".join(
                    [
                        f"{sign}: {', '.join(syns)}"
                        for sign, syns in zip(prediction.split(), synonyms)
                    ]
                )
                st.write(f"Synonyms: {synonyms_text}")

        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
