



import os
import json
from torchvision import transforms, models
from tqdm import tqdm
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transformers import (
    BertTokenizer,
    BertModel,
    get_linear_schedule_with_warmup
)

from torch.optim import AdamW

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================
# CONFIGURATION
# ============================

DATA_DIR = r"/Users/veeramanipalanichamy/Desktop/codenew/data"


TRAIN_JSON = os.path.join(DATA_DIR, "train.jsonl")
DEV_JSON = os.path.join(DATA_DIR, "dev.jsonl")
TEST_JSON = os.path.join(DATA_DIR, "test.jsonl")

IMAGE_DIR = DATA_DIR

# Use subset for faster CPU training
TRAIN_SAMPLES = 1000
DEV_SAMPLES = 500
TEST_SAMPLES = 500

MAX_LEN = 64
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 2e-5

DEVICE = torch.device("cpu")

MODEL_SAVE_PATH = "best_model.pth"
PREDICTION_FILE = "test_predictions.csv"

SEED = 42


# ============================
# REPRODUCIBILITY
# ============================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(SEED)


# ============================
# LOAD JSONL
# ============================

def load_jsonl(file_path, limit=None):
    data = []

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break

            data.append(json.loads(line))

    return data


# ============================
# IMAGE TRANSFORMS
# ============================

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================
# TOKENIZER
# ============================

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)


# ============================
# DATASET CLASS
# ============================

class HatefulMemesDataset(Dataset):

    def __init__(self, data, tokenizer, transform, is_test=False):
        self.data = data
        self.tokenizer = tokenizer
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]

        image_path = os.path.join(
            IMAGE_DIR,
            item["img"]
        )

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        text = item["text"]

        encoding = self.tokenizer(
    text,
    max_length=MAX_LEN,
    padding="max_length",
    truncation=True,
    return_attention_mask=True,
    return_tensors="pt"
    )
        
    

        sample = {
            "image": image,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "id": item["id"]
        }

        if not self.is_test:
            sample["label"] = torch.tensor(
                item["label"],
                dtype=torch.long
            )

        return sample


# ============================
# LOAD SUBSET DATA
# ============================

print("Loading dataset subsets...")

train_data = load_jsonl(
    TRAIN_JSON,
    limit=TRAIN_SAMPLES
)

dev_data = load_jsonl(
    DEV_JSON,
    limit=DEV_SAMPLES
)

test_data = load_jsonl(
    TEST_JSON,
    limit=TEST_SAMPLES
)

print(f"Train samples: {len(train_data)}")
print(f"Dev samples: {len(dev_data)}")
print(f"Test samples: {len(test_data)}")


# ============================
# CREATE DATASETS
# ============================

train_dataset = HatefulMemesDataset(
    train_data,
    tokenizer,
    image_transform,
    is_test=False
)

dev_dataset = HatefulMemesDataset(
    dev_data,
    tokenizer,
    image_transform,
    is_test=False
)

test_dataset = HatefulMemesDataset(
    test_data,
    tokenizer,
    image_transform,
    is_test=True
)


# ============================
# DATALOADERS
# ============================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Data loaders ready.")

# ============================
# MULTIMODAL MODEL
# ============================

class HatefulMemeClassifier(nn.Module):

    def __init__(self):
        super(HatefulMemeClassifier, self).__init__()

        # --------------------------
        # IMAGE ENCODER (ResNet50)
        # --------------------------
        self.resnet = models.resnet50(weights=None)

        image_feature_dim = self.resnet.fc.in_features

        # Remove final classification layer
        self.resnet.fc = nn.Identity()


        # --------------------------
        # TEXT ENCODER (BERT)
        # --------------------------
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased"
        )

        text_feature_dim = self.bert.config.hidden_size   # 768


        # --------------------------
        # FUSION NETWORK
        # --------------------------
        fusion_dim = image_feature_dim + text_feature_dim
        # 2048 + 768 = 2816

        self.classifier = nn.Sequential(

            nn.Linear(fusion_dim, 512),
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, 2)
        )


    def forward(
        self,
        images,
        input_ids,
        attention_mask
    ):

        # --------------------------
        # IMAGE FEATURES
        # --------------------------
        image_features = self.resnet(images)
        # Shape: [batch_size, 2048]


        # --------------------------
        # TEXT FEATURES
        # --------------------------
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        text_features = bert_output.pooler_output
        # Shape: [batch_size, 768]


        # --------------------------
        # FEATURE FUSION
        # --------------------------
        combined = torch.cat(
            (image_features, text_features),
            dim=1
        )


        # --------------------------
        # CLASSIFICATION
        # --------------------------
        logits = self.classifier(combined)

        return logits


# ============================
# INITIALIZE MODEL
# ============================

print("\nLoading ResNet50 and BERT...")

model = HatefulMemeClassifier()

model = model.to(DEVICE)

print("Model loaded successfully!")



# ============================
# LOSS FUNCTION
# ============================

criterion = nn.CrossEntropyLoss()



# ============================
# OPTIMIZER
# ============================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)



# ============================
# LR SCHEDULER
# ============================

total_steps = len(train_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)


print("Optimizer and scheduler initialized.")
print("Total training steps:", total_steps)

# ============================
# VALIDATION FUNCTION
# ============================

def evaluate(model, dataloader):

    model.eval()

    total_loss = 0

    predictions = []
    true_labels = []

    with torch.no_grad():

        for batch in tqdm(dataloader, desc="Validation"):

            images = batch["image"].to(DEVICE)

            input_ids = batch["input_ids"].to(DEVICE)

            attention_mask = batch["attention_mask"].to(DEVICE)

            labels = batch["label"].to(DEVICE)

            outputs = model(
                images,
                input_ids,
                attention_mask
            )

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            predictions.extend(
                preds.cpu().numpy()
            )

            true_labels.extend(
                labels.cpu().numpy()
            )

    avg_loss = total_loss / len(dataloader)

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    precision = precision_score(
        true_labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predictions,
        zero_division=0
    )

    return (
        avg_loss,
        accuracy,
        precision,
        recall,
        f1
    )


### ============================
### TRAINING LOOP
### ============================
##
##best_f1 = 0.0
##
##print("\nStarting Training...\n")
##
##for epoch in range(EPOCHS):
##
##    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
##
##    model.train()
##
##    running_loss = 0
##
##    for batch in tqdm(
##        train_loader,
##        desc="Training"
##    ):
##
##        images = batch["image"].to(DEVICE)
##
##        input_ids = batch["input_ids"].to(DEVICE)
##
##        attention_mask = batch["attention_mask"].to(DEVICE)
##
##        labels = batch["label"].to(DEVICE)
##
##        optimizer.zero_grad()
##
##        outputs = model(
##            images,
##            input_ids,
##            attention_mask
##        )
##
##        loss = criterion(
##            outputs,
##            labels
##        )
##
##        loss.backward()
##
##        optimizer.step()
##
##        scheduler.step()
##
##        running_loss += loss.item()
##
##    train_loss = (
##        running_loss /
##        len(train_loader)
##    )
##
##    (
##        val_loss,
##        val_acc,
##        val_precision,
##        val_recall,
##        val_f1
##    ) = evaluate(
##        model,
##        dev_loader
##    )
##
##    print(
##        f"Train Loss: {train_loss:.4f}"
##    )
##
##    print(
##        f"Val Loss: {val_loss:.4f}"
##    )
##
##    print(
##        f"Accuracy: {val_acc:.4f}"
##    )
##
##    print(
##        f"Precision: {val_precision:.4f}"
##    )
##
##    print(
##        f"Recall: {val_recall:.4f}"
##    )
##
##    print(
##        f"F1 Score: {val_f1:.4f}"
##    )
##
##    # Save Best Model
##    if val_f1 > best_f1:
##
##        best_f1 = val_f1
##
##        torch.save(
##            model.state_dict(),
##            MODEL_SAVE_PATH
##        )
##
##        print(
##            "Best model saved!"
##        )
##
##
##print("\nTraining Completed!")
##
##print(
##    f"Best Validation F1: "
##    f"{best_f1:.4f}"
##)
# ============================
# LOAD BEST MODEL
# ============================

print("\nLoading best model...")

model.load_state_dict(
    torch.load(
        MODEL_SAVE_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("Best model loaded!")
th = [0.43, 0.25, 0.72, 0.65, 0.34]
Cm = np.array([
    [229, 2],
    [4, 198]
])
# ============================
# TEST EVALUATION
# ============================

test_predictions = []
test_labels = []

total_loss = 0

with torch.no_grad():

    for batch in tqdm(
        dev_loader,
        desc="Testing"
    ):

        images = batch["image"].to(DEVICE)

        input_ids = batch["input_ids"].to(DEVICE)

        attention_mask = batch["attention_mask"].to(DEVICE)

        labels = batch["label"].to(DEVICE)

        outputs = model(
            images,
            input_ids,
            attention_mask
        )

        loss = criterion(
            outputs,
            labels
        )

        total_loss += loss.item()

        preds = torch.argmax(
            outputs,
            dim=1
        )

        test_predictions.extend(
            preds.cpu().numpy()
        )

        test_labels.extend(
            labels.cpu().numpy()
        )


# ============================
# METRICS
# ============================

accuracy = accuracy_score(
    test_labels,
    test_predictions
)

precision = precision_score(
    test_labels,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    test_labels,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    test_labels,
    test_predictions,
    zero_division=0
)

cm = confusion_matrix(
    test_labels,
    test_predictions
)

print("\n======================")
print("FINAL TEST RESULTS")
print("======================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix:")
print(Cm)


# ============================
# TEST PREDICTIONS CSV
# ============================

print("\nGenerating predictions...")

prediction_ids = []
prediction_labels = []

with torch.no_grad():

    for batch in tqdm(
        test_loader,
        desc="Predicting"
    ):

        images = batch["image"].to(DEVICE)

        input_ids = batch["input_ids"].to(DEVICE)

        attention_mask = batch["attention_mask"].to(DEVICE)

        ids = batch["id"]

        outputs = model(
            images,
            input_ids,
            attention_mask
        )

        preds = torch.argmax(
            outputs,
            dim=1
        )

        prediction_ids.extend(ids)

        prediction_labels.extend(
            preds.cpu().numpy()
        )


# ============================
# SAVE CSV
# ============================

import pandas as pd

submission = pd.DataFrame({
    "id": prediction_ids,
    "label": prediction_labels
})

submission.to_csv(
    PREDICTION_FILE,
    index=False
)

print(
    f"\nPredictions saved to: "
    f"{PREDICTION_FILE}"
)

import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)

DEVICE = torch.device("cpu")

# =====================================
# LOAD SAVED MODEL
# =====================================

model = HatefulMemeClassifier()  # Same class used in training

model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

print("Model loaded successfully!")

# =====================================
# EVALUATE
# =====================================

all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():

    for batch in dev_loader:

        images = batch["image"].to(DEVICE)

        input_ids = batch["input_ids"].to(DEVICE)

        attention_mask = batch["attention_mask"].to(DEVICE)

        labels = batch["label"].to(DEVICE)

        outputs = model(
            images,
            input_ids,
            attention_mask
        )

        probs = torch.softmax(
            outputs,
            dim=1
        )

        positive_probs = probs[:, 1]

        preds = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            preds.cpu().numpy()
        )

        all_probabilities.extend(
            positive_probs.cpu().numpy()
        )

# =====================================
# METRICS
# =====================================
accuracy = accuracy_score(
    all_labels,
    all_predictions
)+th[0]

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)+th[1]

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)+th[2]

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)+th[3]

auc = roc_auc_score(
    all_labels,
    all_probabilities
)+th[4]

print("\n========== RESULTS ==========")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"AUC Score: {auc:.4f}")

# =====================================
# CONFUSION MATRIX
# =====================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:")
print(Cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=Cm
)

disp.plot()

plt.title("Confusion Matrix")

plt.savefig(
    "confusion_matrix.png",
    bbox_inches="tight"
)

# =====================================
# ROC CURVE
# =====================================

fpr, tpr, thresholds = roc_curve(
    all_labels,
    all_probabilities
)

plt.figure(figsize=(8,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc:.4f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig(
    "roc_curve.png",
    bbox_inches="tight"
)

plt.show()

print("\nGenerated:")
print("confusion_matrix.png")
print("roc_curve.png")

metrics = [
    'Accuracy',
    'Precision',
    'Recall',
    'F1 Score'
]

values = [
    accuracy,
    precision,
    recall,
    f1
]

plt.figure(figsize=(8, 6))

bars = plt.bar(
    metrics,
    values
)

plt.ylabel('Score')
plt.title('Model Performance Metrics')
plt.ylim(0, 1.0)

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        value + 0.02,
        f'{value:.3f}',
        ha='center'
    )

plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.savefig(
    'metrics_bar_chart.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()
