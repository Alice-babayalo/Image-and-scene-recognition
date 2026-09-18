import torch
from torchvision import models, transforms
from PIL import Image
import urllib.request

# 1. Load a pretrained model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# 2. Prepare the image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

img = Image.open("sample.jpg").convert("RGB")
input_tensor = transform(img).unsqueeze(0)

# 3. Run inference
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.nn.functional.softmax(output[0], dim=0)

# 4. Load ImageNet class labels
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = urllib.request.urlopen(url).read().decode("utf-8").splitlines()

# 5. Show the top-3 predictions
top3 = torch.topk(probs, 3)

print("Top-3 predictions:")

for score, idx in zip(top3.values, top3.indices):
    print(f" {labels[idx]:<25} {score.item() * 100:.2f}%")