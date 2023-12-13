import pandas as pd
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
import os
from dataset import DatasetFF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_per_frame_predictions(folders: list[str], model: Module) -> dict[str, float]:
    model.to(device=device)
    model = model.eval()
    predictions: dict[str, Tensor] = dict()
    for folder in folders:
        pd_frame = pd.read_csv(os.path.join(folder, "split.csv"))
        frames = [os.path.join(folder, f) for f in pd_frame["filename"].to_list()]
        labels = pd_frame["label"].to_list()
        dataset = DatasetFF(frames, labels)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)

        predictions_folder: list[float] = list()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                print(f"\r{folder}: [{i+1} / {len(loader)}]          ", end="")
                data = batch[0].to(device)
                pred = model(data)
                pred = torch.flatten(pred).cpu().detach().numpy().tolist()
                predictions_folder.extend(pred)

        for frame, p in zip(frames, predictions_folder):
            predictions[frame] = p

    return predictions

def get_per_video_predictions(predictions: dict[str, float]) -> dict[str, float]:
    files_unique = [key for key in predictions.keys()]
    files_unique = set(["_".join(f.split("_")[:-1]) for f in files_unique])
    predictions_video: dict[str, float] = dict()
    
    for fu in files_unique:
        pred = [value for key, value in predictions.items() if fu in key]
        predictions_video[fu] = sum(pred) / float(len(pred))
    

    keys_sorted = sorted(predictions_video.keys())
    predictions_sorted = {key: predictions_video[key] for key in keys_sorted}
    return predictions_sorted