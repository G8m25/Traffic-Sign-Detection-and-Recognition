import os
import torch
from PIL import Image
import pandas as pd
from torchvision import transforms

# Member 2 integration
from preprocessing import preprocess_pil

def process_img(path, target_format, target_size, augment):
    transform_list = []
    if augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ])
    transform_list.extend([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    transform = transforms.Compose(transform_list)
    img = Image.open(path).convert(target_format)

    # Member 2:
    # denoise + CLAHE contrast enhancement
    img = preprocess_pil(img)
    return transform(img)

class DataLoader:
    def __init__(self, data_dir, lables_dir, batch_size):
        self.data_dir = data_dir
        self.lables_dir = lables_dir
        self.batch_size = batch_size
        self.current_pos = 0

    def __call__(self, format="L", size=(128, 128), augment=True):
        x, y, classes = [], [], []
        df = pd.read_csv(self.lables_dir)
        for _, row in df.iterrows():
            class_id = row["ClassId"]
            classes.append(row["Name"])
            class_path = os.path.join(self.data_dir, str(class_id))
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img_tensor = process_img(img_path, format, size, augment)
                x.append(img_tensor)
                y.append(class_id)
        x = torch.stack(x)
        y = torch.tensor(y)
        return x, y, classes

    def get_batch(self, x, y):
        buffer = torch.arange(self.current_pos, self.current_pos + self.batch_size)
        xb = x[buffer]
        yb = y[buffer]
        self.current_pos += self.batch_size
        if (self.current_pos + self.batch_size) > x.shape[0]:
            self.current_pos = 0
        return xb, yb