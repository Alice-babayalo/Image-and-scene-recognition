import torch
import torchvision
import cv2
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("OpenCV version:", cv2.__version__)
print("GPU available:", torch.cuda.is_available())