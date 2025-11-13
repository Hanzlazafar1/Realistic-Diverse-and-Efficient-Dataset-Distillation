import torch
from torchvision import models, transforms
from PIL import Image

# 1️⃣ Load the student model
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the model architecture (must match the original student model)
student = models.resnet18(num_classes=10)
student.load_state_dict(torch.load(r"C:\Users\Admin\Desktop\REED Data Distilation\exp\cifar10_resnet18_ipc1\syn_data\student_model.pth", map_location=device))
student.to(device)
student.eval()  # set model to evaluation mode

# 2️⃣ Define the image transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10 image size
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # CIFAR-10 mean/std
])

# 3️⃣ Load your test image
img_path = r"C:\Users\Admin\Desktop\REED Data Distilation\cifar10\test\truck\0001.png"  # <-- replace with your image path
img = Image.open(img_path).convert('RGB')  # ensure image is RGB
img_tensor = transform(img).unsqueeze(0).to(device)  # add batch dimension

# 4️⃣ Make prediction
with torch.no_grad():
    outputs = student(img_tensor)
    predicted_class = outputs.argmax(dim=1).item()

# 5️⃣ Map class index to CIFAR-10 labels
cifar10_classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

print(f"Predicted class index: {predicted_class}")
print(f"Predicted label: {cifar10_classes[predicted_class]}")
