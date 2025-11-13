import torch
from torchvision import models, datasets, transforms
from tqdm import tqdm
import os
from utils.utils import set_seed, ensure_dir

def main(args):
    print(f"\n🔹 Starting RDED synthesis on {args.subset}\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    # 1️⃣ Load pretrained teacher model
    teacher = models.resnet18(pretrained=True).to(device)
    teacher.eval()

    # 2️⃣ Create folder to save synthetic data
    ensure_dir(args.syn_data_path)

    # 3️⃣ Load training dataset
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor()
    ])
    train_data = datasets.ImageFolder(root=args.train_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    # 4️⃣ Extract & score patches
    print("🧩 Extracting and scoring patches...")
    class_patches = {}

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            outputs = teacher(images)
            conf, preds = torch.max(torch.softmax(outputs, dim=1), dim=1)

        for img, label, score in zip(images, labels, conf):
            cls = label.item()
            if cls not in class_patches:
                class_patches[cls] = []
            class_patches[cls].append((score.item(), img.cpu()))

    # 5️⃣ Save top-N synthetic images per class
    distilled_images = []
    for cls, patches in class_patches.items():
        patches = sorted(patches, key=lambda x: x[0], reverse=True)
        top_patches = [p[1] for p in patches[:args.ipc]]
        for i, patch in enumerate(top_patches):
            save_path = os.path.join(args.syn_data_path, f"class{cls}_img{i}.pt")
            torch.save(patch, save_path)
            distilled_images.append(save_path)

    print(f"✅ Created {len(distilled_images)} distilled images in {args.syn_data_path}\n")
