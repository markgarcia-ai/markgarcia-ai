import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader

# Toy dataset with 10 images (you'll need to replace this with your dataset)
# Assume image_paths and bounding_boxes are lists containing image paths and bounding box annotations

# ... Load image paths and bounding boxes ...

# Define transforms for data augmentation and normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    # Add more transformations if needed (e.g., data augmentation)
])

# Custom dataset class to load images and annotations
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, bounding_boxes, transform=None):
        self.image_paths = image_paths
        self.bounding_boxes = bounding_boxes
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        boxes = torch.as_tensor(self.bounding_boxes[idx], dtype=torch.float32)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        return image, {
            'boxes': boxes,
            'labels': labels
        }

# Create custom dataset instance
dataset = CustomDataset(image_paths, bounding_boxes, transform=transform)

# Split dataset into train and validation (you may need a more sophisticated split method)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize pre-trained model
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Modify the model to have the correct number of output classes
num_classes = 2  # (background + your object class)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Define optimizer and criterion
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
criterion = torch.nn.SmoothL1Loss()

# Train the model
num_epochs = 10
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {losses.item()}")
