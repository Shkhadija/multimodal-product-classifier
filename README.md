# Multi-Modal Product Classifier (Image + Text)

Classifies e-commerce products by `subCategory` using the product image, its text metadata, or both.
Three models are trained under identical conditions (same stratified split, same epochs, same optimiser, same metric) and compared on one test set.

**Dataset:** [Fashion Product Images (small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)

## Models

| Model | Architecture |
|---|---|
| A: Image-only | ResNet18 (frozen) + linear head |
| B: Text-only | DistilBERT (frozen, mean-pooled) + linear head |
| C: Fusion | Late fusion: ResNet18 (512-d) + DistilBERT (768-d) concatenated, MLP head |

## Results (test set, 1,500 samples, 27 classes)

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| A: Image-only | 0.8700 | 0.6905 | 0.8657 |
| B: Text-only | 0.6967 | 0.5394 | 0.6584 |
| C: Fusion | **0.9140** | **0.7373** | **0.9041** |

Macro F1 is the primary metric because the classes are heavily imbalanced (Topwear is about 35% of the data).

Class-name words (e.g. "Shirts", "Watches") were removed from product titles to prevent label leakage, so the text model is deliberately weaker than with raw titles.

## Setup

- CPU run: 10k stratified subsample, 70/15/15 split, images resized to 112x112, 5 epochs
- Learning rate chosen per model on the validation set (grid: 1e-3, 1e-2)

## Details

Full write-up (class distribution, split strategy, fusion design, error analysis, verdict): see [note.md](note.md).

## Run

```bash
pip install torch torchvision transformers scikit-learn pandas matplotlib kagglehub
jupyter notebook multimodal_product_classifier.ipynb
```
