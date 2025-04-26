import os
from tqdm import tqdm
from create_data import get_data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
from torchvision.models import efficientnet_b3, resnet34, convnext_tiny, alexnet
import torch.optim as optim
import time
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train, tst = get_data()
test, test_indices = tst
val_size = 0.2

end_train_indx = int(len(train) * (1 - val_size))
random.shuffle(train)
train_data, val_data = train[:end_train_indx], train[end_train_indx:]


class CustomDataset(Dataset):
    def __init__(self, data, is_test=False):
        self.data = data
        self.is_test = is_test

    def __getitem__(self, idx):
        if self.is_test:
            return self.data[idx], -1
        return self.data[idx]

    def __len__(self):
        return len(self.data)

train_dataset = CustomDataset(train_data)
val_dataset = CustomDataset(val_data)
test_dataset = CustomDataset(test, is_test=True)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = efficientnet_b3(pretrained=True)

# Модифицируем классификатор
num_classes = 50
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(num_features, num_classes))

model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

num_epochs = 15
best_val_acc = 0.0

for epoch in range(num_epochs):
    print(f'\nEpoch {epoch + 1}/{num_epochs}')
    start_time = time.time()
    model.train()
    running_loss = 0.0
    running_corrects = 0

    # РўСЂРµРЅРёСЂРѕРІРѕС‡РЅС‹Р№ С†РёРєР» СЃ tqdm
    train_loop = tqdm(train_loader, desc='Training', leave=True)
    for inputs, labels in train_loop:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        # РћР±РЅРѕРІР»СЏРµРј РїСЂРѕРіСЂРµСЃСЃ
        train_loop.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{torch.sum(preds == labels.data).item() / inputs.size(0):.2f}"
        })

    train_loss = running_loss / len(train_dataset)
    train_acc = running_corrects.double() / len(train_dataset)

    # Р’Р°Р»РёРґР°С†РёРѕРЅРЅС‹Р№ С†РёРєР» СЃ tqdm
    model.eval()
    val_loss = 0.0
    val_corrects = 0

    val_loop = tqdm(val_loader, desc='Validating', leave=True)
    with torch.no_grad():
        for inputs, labels in val_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)

            val_loop.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{torch.sum(preds == labels.data).item() / inputs.size(0):.2f}"
            })

    val_loss = val_loss / len(val_dataset)
    val_acc = val_corrects.double() / len(val_dataset)

    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f'best_{model._get_name().lower()}.pth')
        end_train_acc = train_acc
        end_val_acc = val_acc

    epoch_time = time.time() - start_time
    print(f'Epoch {epoch + 1}/{num_epochs} | Time: {epoch_time:.2f}s')
    print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}\n')


model.load_state_dict(torch.load(f'best_{model._get_name().lower()}.pth'))
model.eval()

all_preds = []
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())

results = pd.DataFrame({
    'index': test_indices,
    'label': all_preds
}).sort_values('index')

results.to_csv('submission3.csv', index=False)
print(f"Saved predictions for {len(results)} samples to submission.csv")

st = model._get_name().lower() + '\t' + str(end_train_acc) + '\t' + str(end_val_acc) + '\n'


if os.path.exists('results.txt'):
    mode = 'a'
else:
    mode = 'w'

with open('results.txt', mode, encoding='utf-8') as f:
    f.write(st)