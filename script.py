import os
import json
import logging
import yaml
from dataclasses import dataclass, fields
from typing import List, Dict, Optional, Tuple
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ColorJitter
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from google.colab import drive
import requests
import yt_dlp
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class VideoInfo:
    org_text: str
    clean_text: str
    start_time: float
    signer_id: int
    signer: int
    start: int
    end: int
    file: str
    label: int
    height: float
    fps: float
    end_time: float
    url: str
    text: str
    box: List[float]
    width: float
    review: Optional[bool] = None


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_video(
    video_info: VideoInfo, output_path: str, max_retries: int = 3
) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            url = video_info.url
            if "youtube.com" in url or "youtu.be" in url:
                ydl_opts = {
                    "format": "bestvideo[ext=mp4][height<=360]/best[ext=mp4][height<=360]/best[ext=mp4]/best",
                    "outtmpl": output_path,
                    "no_warnings": True,
                    "quiet": True,
                    "no_checkcertificate": True,
                    "nocheckcertificate": True,
                    "ignoreerrors": False,
                    "logtostderr": False,
                    "nopart": True,
                    "no_playlist": True,
                    "postprocessors": [
                        {
                            "key": "FFmpegVideoConvertor",
                            "preferedformat": "mp4",
                        }
                    ],
                    "postprocessor_args": [
                        "-an",
                        "-vcodec",
                        "libx264",
                        "-preset",
                        "ultrafast",
                        "-crf",
                        "23",
                    ],
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
            else:
                if not url.startswith(("http://", "https://")):
                    url = "https://" + url
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            return output_path
        except Exception as e:
            logging.warning(
                f"Attempt {attempt + 1} failed to download video from {url}: {str(e)}"
            )
            time.sleep(1)

    logging.error(f"Failed to download video after {max_retries} attempts: {url}")
    return None


def trim_video(
    input_path: str, start_time: float, end_time: float, output_path: str
) -> Optional[str]:
    try:
        cmd = [
            "ffmpeg",
            "-i",
            input_path,
            "-ss",
            str(start_time),
            "-to",
            str(end_time),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-an",
            "-y",
            output_path,
        ]
        subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        os.remove(input_path)
        return output_path
    except Exception as e:
        logging.error(f"Failed to process video {input_path}: {str(e)}")
        if os.path.exists(input_path):
            os.remove(input_path)
        return None


def is_valid_video(file_path: str) -> bool:
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        if not ret:
            return False
        cap.release()
        return True
    except Exception as e:
        logging.error(f"Error validating video {file_path}: {str(e)}")
        return False


def process_video(
    video_info: VideoInfo, temp_dir: str, final_dir: str
) -> Optional[str]:
    signer = video_info.signer_id if video_info.signer_id != -1 else video_info.signer
    output_filename = f"{video_info.clean_text}_{signer}.mp4"
    final_path = os.path.join(final_dir, output_filename)

    if os.path.exists(final_path):
        logging.info(f"Video {output_filename} already exists, skipping...")
        return final_path

    temp_filename = f"temp_{video_info.clean_text}_{signer}.mp4"
    temp_path = os.path.join(temp_dir, temp_filename)

    downloaded_path = download_video(video_info, temp_path)
    if downloaded_path and is_valid_video(downloaded_path):
        return trim_video(
            downloaded_path, video_info.start_time, video_info.end_time, final_path
        )
    else:
        logging.error(f"Invalid or corrupted video: {video_info.url}")
        if downloaded_path and os.path.exists(downloaded_path):
            os.remove(downloaded_path)
        return None


def download_and_process_videos(data_dir: str, config: Dict) -> Tuple[int, int]:
    temp_dir = "/content/temp_videos"
    os.makedirs(temp_dir, exist_ok=True)

    final_dir = os.path.join("/content/drive/MyDrive", config["drive_video_folder"])
    os.makedirs(final_dir, exist_ok=True)

    all_videos = []
    for split in ["train", "test", "val"]:
        json_file_path = os.path.join(data_dir, f"MSASL_{split}.json")
        if os.path.exists(json_file_path):
            with open(json_file_path, "r") as f:
                videos = json.load(f)
                for v in videos:
                    if isinstance(v["box"], str):
                        v["box"] = eval(v["box"])
                    video_info_fields = set(field.name for field in fields(VideoInfo))
                    filtered_v = {k: v[k] for k in video_info_fields if k in v}
                    all_videos.append(VideoInfo(**filtered_v))
        else:
            logging.warning(f"File {json_file_path} not found. Skipping...")

    logging.info(f"Total videos available: {len(all_videos)}")

    class_videos = {}
    for video in all_videos:
        if video.label not in class_videos:
            class_videos[video.label] = []
        class_videos[video.label].append(video)

    processed_labels = set()
    total_classes = len(class_videos)

    def process_class(label):
        videos = class_videos[label]
        for video_info in videos:
            output_filename = f"{video_info.clean_text}_{video_info.signer_id if video_info.signer_id != -1 else video_info.signer}.mp4"
            final_path = os.path.join(final_dir, output_filename)
            if os.path.exists(final_path):
                logging.info(f"Video for class {label} already exists, skipping...")
                return label
            result = process_video(video_info, temp_dir, final_dir)
            if result:
                logging.info(f"Successfully processed video for class {label}")
                return label
        return None

    with ThreadPoolExecutor(max_workers=config.get("max_workers", 4)) as executor:
        futures = [
            executor.submit(process_class, label) for label in class_videos.keys()
        ]
        for future in tqdm(
            as_completed(futures), total=total_classes, desc="Processing classes"
        ):
            processed_label = future.result()
            if processed_label is not None:
                processed_labels.add(processed_label)

    logging.info(
        f"Successfully processed videos for {len(processed_labels)} out of {total_classes} classes"
    )

    return len(processed_labels), total_classes


class MSASLDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        config: Dict,
        transform: Optional[transforms.Compose] = None,
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.video_dir = os.path.join(
            "/content/drive/MyDrive", config["drive_video_folder"]
        )

        with open(os.path.join(data_dir, "MSASL_classes.json"), "r") as f:
            self.classes = json.load(f)

        self.data = []
        for split in ["train", "val", "test"]:
            split_path = os.path.join(data_dir, f"MSASL_{split}.json")
            if os.path.exists(split_path):
                with open(split_path, "r") as f:
                    all_data = json.load(f)
                    for item in all_data:
                        video_path = os.path.join(
                            self.video_dir,
                            f"{item['clean_text']}_{item['signer_id'] if item['signer_id'] != -1 else item['signer']}.mp4",
                        )
                        item["video_path"] = video_path
                        self.data.append(item)

        available_labels = set(item["label"] for item in self.data)
        self.class_to_idx = {
            label: idx for idx, label in enumerate(sorted(available_labels))
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        label = self.class_to_idx[item["label"]]
        video_path = item["video_path"]

        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                raise Exception("Failed to read frame")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)

        frame = cv2.resize(frame, (224, 224))

        if self.transform:
            frame = self.transform(frame)

        return frame, label


def get_data_loaders(data_dir: str, config: Dict):
    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            RandomHorizontalFlip(),
            RandomRotation(10),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_test_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    full_dataset = MSASLDataset(data_dir, config, transform=None)

    train_size = int(config["train_ratio"] * len(full_dataset))
    val_size = int(config["val_ratio"] * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size + test_size]
    )
    val_dataset, test_dataset = torch.utils.data.random_split(
        val_test_dataset, [val_size, test_size]
    )

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.transform = val_test_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    logging.info(f"Train set size: {len(train_dataset)}")
    logging.info(f"Validation set size: {len(val_dataset)}")
    logging.info(f"Test set size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, full_dataset


class SignLanguageModel(nn.Module):
    def __init__(self, num_classes: int):
        super(SignLanguageModel, self).__init__()

        self.base_model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

        for param in list(self.base_model.parameters())[:-30]:
            param.requires_grad = False

        num_ftrs = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

    def to_torchscript(self, example_input):
        self.eval()
        traced_script_module = torch.jit.trace(self, example_input)
        return traced_script_module


class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def train_model(data_dir: str, config: Dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    train_loader, val_loader, test_loader, full_dataset = get_data_loaders(
        data_dir, config
    )

    num_classes = len(full_dataset.class_to_idx)
    logging.info(f"Number of classes: {num_classes}")

    model = SignLanguageModel(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.5
    )
    early_stopping = EarlyStopping(patience=10)
    scaler = GradScaler()

    best_val_accuracy = 0.0

    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"
        ):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_accuracy = 100 * train_correct / train_total
        logging.info(
            f"Epoch {epoch+1}/{config['num_epochs']}, Training Loss: {train_loss/len(train_loader):.4f}, Training Accuracy: {train_accuracy:.2f}%"
        )

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        logging.info(
            f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%"
        )

        scheduler.step(val_loss)
        early_stopping(val_loss)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), config["model_save_path"])
            logging.info("New best model saved!")

        if early_stopping.early_stop:
            logging.info("Early stopping")
            break

    logging.info("Training completed. Evaluating on test set...")

    model.load_state_dict(torch.load(config["model_save_path"]))
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = 100 * test_correct / test_total
    logging.info(f"Test Accuracy: {test_accuracy:.2f}%")
    logging.info(f"Best model saved to {config['model_save_path']}")

    example_input = torch.rand(1, 3, 224, 224).to(device)
    torchscript_model = model.to_torchscript(example_input)
    torch.jit.save(torchscript_model, config["torchscript_model_save_path"])
    logging.info(f"TorchScript model saved to {config['torchscript_model_save_path']}")


def main():
    try:
        drive.mount("/content/drive")
    except:
        print("Drive already mounted")

    config_path = "/content/drive/MyDrive/Task2/config.yaml"
    config = load_config(config_path)

    logging.info("Starting video download and processing...")
    processed_classes, total_classes = download_and_process_videos(
        config["data_dir"], config
    )
    logging.info(f"Processed {processed_classes} out of {total_classes} classes")

    logging.info("Starting model training...")
    train_model(config["data_dir"], config)


if __name__ == "__main__":
    main()
