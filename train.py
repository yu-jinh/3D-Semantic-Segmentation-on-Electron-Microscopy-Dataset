from data_preparing.load_data import create_dataset
import os
from datasets import Dataset, DatasetDict
from transformers import AutoImageProcessor
import numpy as np
import torch
from torch import nn
import evaluate
from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer
from torchvision.transforms import ColorJitter

def main():
    train_path, train_label_path = os.path.join("data","training.tif"), os.path.join("data","training_groundtruth.tif")
    test_path, test_label_path = os.path.join("data", "testing.tif"), os.path.join("data","testing_groundtruth.tif")

    train_dataset = create_dataset(train_path, train_label_path)
    test_dataset = create_dataset(test_path, test_label_path)

    sample = np.array(train_dataset[0]['label'])
    print(sample.shape)



    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    id2label = {0: 'background', 1: 'target'}
    label2id = {'background':0, 'target': 1}
    num_labels = len(id2label)

    checkpoint = "nvidia/mit-b0"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint, reduce_labels=True)

    jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
    def transforms(example_batch):
        print(example_batch)
        images = [x.convert("RGB") for x in example_batch["image"]]
        labels = [x for x in example_batch["label"]]
        inputs = image_processor(images, labels)
        return inputs

    train_dataset.set_transform(transforms)
    test_dataset.set_transform(transforms)

    print(train_dataset[0])

    print("done tranformation")


    metric = evaluate.load("mean_iou")

    def compute_metrics(eval_pred):
        with torch.no_grad():
            logits, labels = eval_pred
            logits_tensor = torch.from_numpy(logits)
            logits_tensor = nn.functional.interpolate(
                logits_tensor,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)

            pred_labels = logits_tensor.detach().cpu().numpy()
            metrics = metric.compute(
                predictions=pred_labels,
                references=labels,
                num_labels=num_labels,
                ignore_index=255,
                reduce_labels=False,
            )
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    metrics[key] = value.tolist()
            return metrics

    # from datasets import load_dataset

    # ds = load_dataset("scene_parse_150", split="train[:50]")

    # ds = ds.train_test_split(test_size=0.2)
    # train_ds = ds["train"]
    # test_ds = ds["test"]

    model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint, id2label=id2label, label2id=label2id)

    training_args = TrainingArguments(
        output_dir="Electron-Microscopy-seg",
        learning_rate=6e-5,
        num_train_epochs=50,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        save_total_limit=3,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=20,
        eval_steps=20,
        logging_steps=1,
        eval_accumulation_steps=5,
        remove_unused_columns=False,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    data_loader = trainer.get_train_dataloader()

    for data in enumerate(data_loader):
        print(data)
        break

    # trainer.train()

if __name__ == "__main__":
    main()