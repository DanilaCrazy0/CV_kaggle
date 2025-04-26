
# Классификация бабочек (Kaggle Competition)

Проект для соревнования Kaggle https://www.kaggle.com/competitions/butterflies-classification/overview) с классификацией 50 видов бабочек.


1. **Установка зависимостей**:
```bash
pip install torch torchvision pandas tqdm
```

2. Скачайте данные с Kaggle и разместите в папках:

/train_butterflies/train_split/class_0...49/
/test_butterflies/valid/


3. **Запуск обучения**:
```bash
python net.py
```

## 📁 Структура проекта

```
butterfly-classification/
├── create_data.py       # Загрузка и аугментация данных
├── net.py               # Модель и обучение
├── best_efficientnet.pth # Веса модели
├── submission.csv       # Предсказания
└── results.txt          # Метрики
```

## 🔧 Технические детали

**Модель**: 
- EfficientNet-B3 (transfer learning)
- Дообученный классификатор на 50 классов
- Аугментация: случайные повороты, отражения, цветовые искажения

**Обучение**:
- 15 эпох
- SGD с моментом (lr=0.001)
- StepLR scheduler (gamma=0.1)
- Batch size: 32

## 📌 Пример использования

```python
model = efficientnet_b3(pretrained=False)
model.load_state_dict(torch.load('best_efficientnet.pth'))
model.eval()
```
