import os
import logging
import yaml
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import cv2
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import yt_dlp
import subprocess
import random
import hashlib
import torch.utils.checkpoint as checkpoint
from torchvision.transforms import v2

torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@dataclass
class VideoInfo:
    org_text: str
    clean_text: str
    start_time: float
    end_time: float
    signer_id: int
    signer: int
    start: int
    end: int
    file: str
    label: int
    height: float
    fps: float
    url: str
    text: str
    box: List[float]
    width: float


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    base_dir = os.getcwd()
    for key in ["data_dir", "processed_video_folder", "model_save_path", "cache_dir"]:
        if key in config:
            config[key] = os.path.join(base_dir, config[key])
    return config


def get_video_hash(video_info):
    return hashlib.md5(
        f"{video_info.clean_text}_{video_info.signer_id}_{video_info.start_time}_{video_info.end_time}".encode()
    ).hexdigest()


def download_video(url, output_path):
    ydl_opts = {
        "format": "bestvideo[height<=240][ext=mp4]",
        "outtmpl": output_path,
        "no_warnings": True,
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
            return True
        except Exception as e:
            logging.error(f"Error downloading video: {str(e)}")
            return False


def process_video(input_path, output_path, start_time, end_time):
    try:
        trim_command = [
            "ffmpeg",
            "-i",
            input_path,
            "-ss",
            str(start_time),
            "-to",
            str(end_time),
            "-c:v",
            "libx264",
            "-an",
            "-vf",
            "scale=224:224",
            "-y",
            output_path,
        ]
        subprocess.run(
            trim_command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error processing video: {str(e)}")
        return False


def download_and_process_video(video_info: VideoInfo, output_dir: str, cache_dir: str):
    try:
        video_hash = get_video_hash(video_info)
        cache_path = os.path.join(cache_dir, f"{video_hash}.mp4")
        processed_output_path = os.path.join(
            output_dir, f"{video_info.clean_text}_{video_info.signer_id}.mp4"
        )

        if os.path.exists(cache_path):
            logging.info(f"Using cached video: {cache_path}")
            os.makedirs(os.path.dirname(processed_output_path), exist_ok=True)
            os.replace(cache_path, processed_output_path)
            return processed_output_path

        raw_output_path = os.path.join(cache_dir, f"{video_hash}_raw.mp4")

        os.makedirs(os.path.dirname(raw_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(processed_output_path), exist_ok=True)

        if not download_video(video_info.url, raw_output_path):
            return None

        if process_video(
            raw_output_path,
            processed_output_path,
            video_info.start_time,
            video_info.end_time,
        ):
            os.remove(raw_output_path)
            return processed_output_path
        else:
            os.remove(raw_output_path)
            return None

    except Exception as e:
        logging.error(f"Failed to download/process video: {str(e)}")
        if os.path.exists(raw_output_path):
            os.remove(raw_output_path)
        return None


def download_and_process_videos(
    data_dir: str,
    output_dir: str,
    cache_dir: str,
    num_workers: int = 4,
    percent: float = 0.1,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    all_videos = []
    for split in ["train", "test", "val"]:
        json_file = os.path.join(data_dir, f"MSASL_{split}.json")
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                videos = json.load(f)
                for v in videos:
                    video_info = VideoInfo(
                        org_text=v["org_text"],
                        clean_text=v["clean_text"],
                        start_time=v["start_time"],
                        end_time=v["end_time"],
                        signer_id=v["signer_id"],
                        signer=v["signer"],
                        start=v["start"],
                        end=v["end"],
                        file=v["file"],
                        label=v["label"],
                        height=v["height"],
                        fps=v["fps"],
                        url=v["url"],
                        text=v["text"],
                        box=v["box"],
                        width=v["width"],
                    )
                    all_videos.append(video_info)

    target_count = max(1, int(len(all_videos) * percent))
    selected_videos = random.sample(all_videos, min(target_count, len(all_videos)))

    processed_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_video = {
            executor.submit(
                download_and_process_video, video, output_dir, cache_dir
            ): video
            for video in selected_videos
        }
        for future in tqdm(
            as_completed(future_to_video),
            total=len(selected_videos),
            desc="Downloading and processing videos",
        ):
            video = future_to_video[future]
            try:
                path = future.result()
                if path:
                    logging.info(f"Processed: {path}")
                    processed_count += 1
                else:
                    logging.warning(
                        f"Failed to process video: {video.clean_text}_{video.signer_id}"
                    )
                    error_count += 1
            except Exception as e:
                logging.error(f"Error processing video: {str(e)}")
                error_count += 1

    logging.info(
        f"Number of unique classes: {len(set(v.clean_text for v in all_videos))}"
    )
    logging.info(f"Total number of videos selected: {len(selected_videos)}")
    logging.info(f"Number of videos successfully processed: {processed_count}")
    logging.info(f"Number of videos that failed to process: {error_count}")


def to_device(x, device):
    return torch.from_numpy(x).to(device)


def normalize(x):
    return x.float() / 255.0


class MSASLDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        processed_video_folder: str,
        split: str,
        transform: transforms.Compose = None,
        sequence_length: int = 12,
        image_size: int = 224,
    ):
        self.image_size = image_size
        self.data_dir = data_dir
        self.processed_video_folder = processed_video_folder
        self.sequence_length = sequence_length

        with open(os.path.join(data_dir, "MSASL_classes.json"), "r") as f:
            self.classes = json.load(f)

        if isinstance(self.classes[0], dict):
            self.class_to_idx = {
                item["clean_text"]: idx for idx, item in enumerate(self.classes)
            }
            self.classes = [item["clean_text"] for item in self.classes]
        else:
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        with open(os.path.join(data_dir, f"MSASL_{split}.json"), "r") as f:
            self.data = json.load(f)

        self.data = [
            item
            for item in self.data
            if item["clean_text"] in self.class_to_idx
            and os.path.exists(
                os.path.join(
                    self.processed_video_folder,
                    f"{item['clean_text']}_{item['signer_id']}.mp4",
                )
            )
        ]

        logging.info(f"Number of valid items in {split} dataset: {len(self.data)}")

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        synonym_path = os.path.join(data_dir, "MSASL_synonym.json")
        if os.path.exists(synonym_path):
            with open(synonym_path, "r") as f:
                self.synonyms = json.load(f)
            if isinstance(self.synonyms, list):
                self.synonyms = {
                    item["clean_text"]: item.get("synonyms", [])
                    for item in self.synonyms
                    if isinstance(item, dict) and "clean_text" in item
                }
        else:
            self.synonyms = {}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        video_path = os.path.join(
            self.processed_video_folder, f"{item['clean_text']}_{item['signer_id']}.mp4"
        )

        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if frame_count <= 0:
                raise ValueError(f"Invalid or empty video file: {video_path}")

            for _ in range(self.sequence_length):
                frame_idx = (
                    np.random.randint(0, frame_count - 1) if frame_count > 1 else 0
                )
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    raise ValueError(f"Failed to read frame from video: {video_path}")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.image_size, self.image_size))
                frames.append(frame)

            cap.release()

            frames = torch.from_numpy(np.stack(frames)).float()
            frames = frames.permute(0, 3, 1, 2)
            frames = frames / 255.0

            if self.transform:
                frames = self.transform(frames)

            label = torch.tensor(self.class_to_idx[item["clean_text"]])

            return frames, label

        except Exception as e:
            logging.error(f"Error processing item {idx}: {str(e)}")
            placeholder_frames = torch.zeros(
                (self.sequence_length, 3, self.image_size, self.image_size)
            )
            placeholder_label = torch.tensor(0)
            return placeholder_frames, placeholder_label

    def get_class_name(self, idx: int) -> str:
        return self.idx_to_class[idx]

    def get_synonyms(self, sign: str) -> List[str]:
        return self.synonyms.get(sign, [])


class SignLanguageModel(nn.Module):
    def __init__(self, num_classes: int, sequence_length: int):
        super(SignLanguageModel, self).__init__()
        self.base_model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        self.base_model.classifier = nn.Identity()

        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            features = self.base_model(dummy_input)
            self.num_features = features.shape[1]

        self.lstm = nn.LSTM(
            self.num_features, 128, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)

        def custom_forward(x):
            return self.base_model(x)

        features = checkpoint.checkpoint(custom_forward, x, use_reentrant=False)
        features = features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(features)
        out = self.fc(lstm_out[:, -1, :])
        return out


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    frames, labels = zip(*batch)
    frames = torch.stack(frames)
    labels = torch.tensor(labels)
    return frames, labels


def train_model(config):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = MSASLDataset(
        config["data_dir"],
        config["processed_video_folder"],
        "train",
        transform,
        config["sequence_length"],
        config["image_size"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=config["prefetch_factor"] if torch.cuda.is_available() else 2,
        collate_fn=collate_fn,
    )

    model = SignLanguageModel(len(train_dataset.classes), config["sequence_length"]).to(
        device
    )

    if torch.cuda.is_available():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=1e-2
    )

    steps_per_epoch = min(
        len(train_dataset) // config["batch_size"],
        config.get("max_steps_per_epoch", 200),
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["learning_rate"],
        steps_per_epoch=steps_per_epoch // config["gradient_accumulation_steps"],
        epochs=config["num_epochs"],
        pct_start=0.3,
    )

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        progress_bar = tqdm(
            enumerate(train_loader),
            total=steps_per_epoch,
            desc=f"Epoch {epoch+1}/{config['num_epochs']}",
        )

        for i, (sequences, labels) in progress_bar:
            if i >= steps_per_epoch:
                break

            sequences = sequences.to(device)
            labels = labels.to(device)

            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)
                    loss = loss / config["gradient_accumulation_steps"]
                scaler.scale(loss).backward()
            else:
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                loss = loss / config["gradient_accumulation_steps"]
                loss.backward()

            if (i + 1) % config["gradient_accumulation_steps"] == 0:
                if torch.cuda.is_available():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            train_loss += loss.item() * config["gradient_accumulation_steps"]
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

            progress_bar.set_postfix(
                {
                    "loss": f"{train_loss / (i + 1):.4f}",
                    "accuracy": f"{correct_predictions / total_predictions:.4f}",
                }
            )

        avg_loss = train_loss / steps_per_epoch
        accuracy = correct_predictions / total_predictions
        logging.info(
            f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
        )

    torch.save(model.state_dict(), config["model_save_path"])
    logging.info(f"Model saved to {config['model_save_path']}")


def main():
    logging.info("Starting Sign Language Interpreter application")
    config = load_config("config.yaml")
    logging.info(f"Loaded configuration from config.yaml: {config}")

    logging.info(
        f"Starting video download and processing ({config['percent']}% of dataset)..."
    )
    download_and_process_videos(
        config["data_dir"],
        config["processed_video_folder"],
        config["cache_dir"],
        config["num_workers"],
        config["percent"],
    )
    config["num_workers"] = (
        config["num_workers_cuda"]
        if torch.cuda.is_available()
        else config["num_workers_cpu"]
    )

    logging.info("Starting model training...")
    train_model(config)
    logging.info("Model training completed")
    logging.info("Sign Language Interpreter application finished")


if __name__ == "__main__":
    main()
