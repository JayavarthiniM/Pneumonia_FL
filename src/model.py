import torch
import torch.nn as nn
from torchvision import models


def create_model():

    model = models.densenet121(
        weights=models.DenseNet121_Weights.DEFAULT
    )

    # Replace the final classifier
    num_features = model.classifier.in_features

    model.classifier = nn.Linear(
        num_features,
        2
    )

    return model


if __name__ == "__main__":

    model = create_model()

    print(model.classifier)

    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)