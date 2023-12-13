import torch
import torch.nn as nn
from torchvision.models import resnet50

def get_resnet50(model_weights: str = None) -> nn.Module:
    if model_weights:
        model = resnet50()
        model.fc = nn.Linear(in_features=2048, out_features=1)
        model.load_state_dict(torch.load(model_weights))
        return model

    repo = 'pytorch/vision'
    model = torch.hub.load(repo, 'resnet50', weights='ResNet50_Weights.IMAGENET1K_V2')
    model.fc = nn.Linear(in_features=2048, out_features=1)
    return model


if __name__ == "__main__":
    model = get_resnet50()

    print(model)