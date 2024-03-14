from PIL import Image
import os
import numpy as np
from datasets import Dataset, dataset_dict

def read_tif(path):
    images = []
    with Image.open(path) as img:
        for i in range(img.n_frames):
            img.seek(i)
            images.append(img.copy())
        images = np.array(images)
    return images

def create_dataset(image_path, label_path):
    images = read_tif(image_path)
    labels = read_tif(label_path)
    dataset = Dataset.from_dict({"image": images,
                                "annotation": labels})
    return dataset 

# path = os.path.join("data", "training.tif")
# images = read_tif(path)
# print(images.shape)