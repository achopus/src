import os
import torch
import numpy as np
from torch import Tensor
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2


def get_files(folder_real: str, folders_fake: list[str] = None, folders_both: list[int] = None, folders_train: list[int] = None, folders_valid: list[int] = None) -> list[list[str]]:
    """_summary_

    Args:
        folder_real (str): _description_
        folders (list[str], optional): _description_. Defaults to None.
        folders_train (list[str], optional): _description_. Defaults to None.
        folders_valid (list[str], optional): _description_. Defaults to None.

    Returns:
        list[list[str]]: _description_
    """

    assert (folders_both or folders_train) or (folders_both or folders_valid), "At least one folder must assign to each train and valid or to both."
    if folders_train is not None and folders_valid is not None:
        assert len(set(folders_train).intersection(folders_valid)) == 0, "No overlap allowed between train and valid dataset, use `folders` parameter to use in both."

    np.random.seed(0)

    # Set None to empty list
    if not folders_train: folders_train = list()
    if not folders_valid: folders_valid = list()

    split_real = pd.read_csv(os.path.join(folder_real, "split.csv"))
    split_real = split_real.applymap(lambda x: os.path.join(folder_real, x) if type(x) == str else x)

    split_fake = pd.DataFrame(columns=["filename", "mode", "label"])
    for f in folders_fake:

        if f in folders_train:
            df = pd.read_csv(os.path.join(f, "split.csv"))
            df["mode"] = 0
        elif f in folders_valid:
            df = pd.read_csv(os.path.join(f, "split_valid.csv"))
        elif f in folders_both:
            df = pd.read_csv(os.path.join(f, "split.csv"))

        df = df.applymap(lambda x: os.path.join(f, x) if type(x) == str else x)
        split_fake = split_fake._append(df)


    train_real = split_real[split_real["mode"] == 0]
    valid_real = split_real[split_real["mode"] == 1]
    test_real = split_real[split_real["mode"] == 2]

    train_fake = split_fake[split_fake["mode"] == 0]
    valid_fake = split_fake[split_fake["mode"] == 1]
    test_fake = split_fake[split_fake["mode"] == 2]

    def combine(real: pd.DataFrame, fake: pd.DataFrame) -> pd.DataFrame:
        n_real = len(real)
        n_fake = len(fake)
        N = min(n_real, n_fake)
        real = real.sample(N)
        fake = fake.sample(N)
        return real._append(fake) 

    train = combine(train_real, train_fake)
    valid = combine(valid_real, valid_fake)
    test = combine(test_real, test_fake)

    train_files = train["filename"].to_list()
    valid_files = valid["filename"].to_list()
    test_files = test["filename"].to_list()
    train_labels = train["label"].to_list()
    valid_labels = valid["label"].to_list()
    test_labels = test["label"].to_list()

    return train_files, train_labels, valid_files, valid_labels, test_files, test_labels


class DatasetFF(Dataset):
    def __init__(self, files: list[str], labels: list[int], transform = None) -> None:
        self.files = files
        self.labels = labels
        self.N = len(self.files)
        self.T = transform

    def __getitem__(self, index) -> tuple[Tensor, float]:
        image = cv2.imread(self.files[index])
        label = self.labels[index]

        image = np.flip(image, axis=-1) / 255
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image) if not self.T else self.T(image)

        return image.float(), torch.tensor(label).float()

    def __len__(self) -> int:
        return self.N


def get_dataloaders(folder_real: str, folders_fake: list[str],
                    folders_both: list[str] = None, folders_train: list[str] = None, folders_valid: list[str] = None,
                    batch_size: int = 64, transform = None) -> tuple[DataLoader, DataLoader, DataLoader]:

    F = get_files(folder_real, folders_fake, folders_both, folders_train, folders_valid)
    train_files, train_labels, valid_files, valid_labels, test_files, test_labels  = F

    dataset_train = DatasetFF(train_files, train_labels, transform)
    dataset_valid = DatasetFF(valid_files, valid_labels)
    dataset_test = DatasetFF(test_files, test_labels)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

if __name__ == "__main__":
    folder_real = "faces/real/"
    folders_fake = ["faces/fake/FaceSwap", "faces/fake/NeuralTextures"]

    folders_both = []
    folders_train = ["faces/fake/FaceSwap"]
    folders_valid = ["faces/fake/NeuralTextures"]

    tr_l, vl_l, ts_l = get_dataloaders(folder_real, folders_fake, folders_both, folders_train, folders_valid)

    print()