import torch
from torch import nn, optim
from torchvision import models
from tqdm import tqdm
import os

def main(args):
    print(f"🔹 Starting student training (validation) on {args.subset}\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1️⃣ Load distilled synthetic dataset
    synthetic_images, labels = [], []
    for file in os.listdir(args.syn_data_path):
        if file.endswith(".pt"):
            tensor = torch.load(os.path.join(args.syn_data_path, file))
            synthetic_images.append(tensor)
            cls_id = int(file.split("_")[0].replace("class", ""))
            labels.append(cls_id)

    X = torch.stack(synthetic_images).to(device)
    y = torch.tensor(labels).to(device)

    # 2️⃣ Load student model
    student = models.resnet18(num_classes=args.nclass).to(device)

    # 3️⃣ Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(student.parameters(), lr=args.adamw_lr)

    # 4️⃣ Train student on synthetic data
    print("🚀 Training student model...\n")
    for epoch in range(args.re_epochs):
        student.train()
        optimizer.zero_grad()
        outputs = student(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{args.re_epochs}] Loss={loss.item():.4f}")

    # 5️⃣ Save trained student model
    model_path = os.path.join(args.syn_data_path, "student_model.pth")
    torch.save(student.state_dict(), model_path)
    print(f"\n✅ Training finished. Model saved at {model_path}\n")
