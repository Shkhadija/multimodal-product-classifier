# Multi-Modal Product Classifier: Notes

Task: classify products by `subCategory` from the image, the text, or both. Dataset: [Fashion Product Images (small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small). CPU only, so a 10k subsample and 112x112 images were used. All numbers come from the notebook outputs.

## 1. Data audit and class distribution

- `styles.csv` read with `on_bad_lines="skip"`: 44,424 rows (number of skipped malformed lines was not measured).
- 5 rows without an image were dropped, leaving 44,419. Missing text fields were filled with an empty string.
- Level chosen: **`subCategory`** (45 classes). Strongly imbalanced: Topwear 15,398 (34.67%), Shoes 7,343 (16.53%), Bags 3,055 (6.88%), Bottomwear 2,693 (6.06%), Watches 2,542 (5.72%). Plot: `class_distribution.png`.
- **Rare classes were dropped** (not merged): 10 classes with fewer than 20 samples (63 rows), then 8 more with fewer than 100 (382 rows), so every class survives a stratified 10k subsample. Final: **27 classes, 43,974 rows**. The smallest test class still has only 3 samples, so its per-class F1 is noisy.

**Compute:** training was done on CPU only (about 80-140 s per epoch for the image and fusion models on 7,000 training samples). Following the task's compute note, a stratified 10k subsample of the 43,974 rows was used instead of the full dataset, with 112x112 images and 5 epochs. Training time on the full data was not measured. Consequences: results may differ on the full dataset, and rare classes become very small in the subsample (the smallest test class has 3 samples).

## 2. Split

10,000 rows sampled with stratification (seed 42), then a stratified **70/15/15 split = 7,000 / 1,500 / 1,500**. The same split was used by all three models. For each model the best epoch and the learning rate were chosen on validation macro F1; test metrics were also printed for every candidate run but not used for selection.

## 3. Preprocessing

- **Image:** resize to 112x112, ImageNet normalisation. Augmentation on train only: horizontal flip, rotation (10 degrees), colour jitter.
- **Text:** `productDisplayName` + `baseColour` + `usage`, DistilBERT tokenizer, padded/truncated to 32 tokens.
- **Leakage masking:** titles often contain the class name ("Shirts", "Shoe"). Words from the dataset's category names were removed from titles for all rows by the same rule (e.g. `Skechers Men Regions Black Shoe` -> `Skechers Men Regions Black`). It can also remove non-label words. The unmasked-title experiment was **not run**.

## 4. Models

| Model | Architecture |
|---|---|
| A: Image-only | ResNet18 (frozen) + `Linear(512, 27)` |
| B: Text-only | DistilBERT (frozen, mean-pooled 768-d) + `Linear(768, 27)` |
| C: Fusion | ResNet18 512-d + DistilBERT 768-d concatenated -> `Linear(1280,256)` -> ReLU -> Dropout(0.3) -> `Linear(256,27)` |

Same setup for all: CrossEntropyLoss (no class weights), AdamW, batch 64, 5 epochs, best epoch by validation macro F1, one run each.

**Fusion type:** early fusion combines modalities at the input/low-level stage; late fusion processes each modality with its own encoder and combines them near the end; joint fusion also trains the encoders end-to-end. **I built late fusion by concatenation.** Both encoders are **frozen** and only the head is trained, so it is not joint training. Fine-tuning/unfreezing was not tried.

**Learning rate** (grid 1e-3, 1e-2, chosen by validation macro F1): image 1e-2 (0.6618 vs 0.6138), text 1e-2 (0.5618 vs 0.3298), fusion 1e-3 (0.7371 vs 0.4964). Fusion at 1e-2 reached only 0.5042 test macro F1, so it is LR-sensitive.

## 5. Results (test, 1,500 samples)

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| A: Image-only | 0.8700 | 0.6905 | 0.8657 |
| B: Text-only | 0.6967 | 0.5394 | 0.6584 |
| C: Fusion | **0.9140** | **0.7373** | **0.9041** |

**Why macro F1:** it weights every class equally, so predicting only the big classes cannot score well. With Topwear at 34.67%, accuracy flatters the models (image-only: accuracy 0.87 but macro F1 0.69).

Plots: `training_curves.png`, `confusion_matrix.png`, `per_class_f1.png`.

**Worst three classes (fusion F1):** Apparel Set 0.00 (3 test samples), Free Gifts 0.00 (3 samples), Dress 0.10 (17 samples). The first two are unreliable; Dress is the real weak class.

**Fusion vs text-only per class:** biggest gains Bottomwear +0.80, Belts +0.71, Wallets +0.54, Eyewear +0.52, Sandal +0.50; biggest losses Free Gifts -0.40 (3 samples), Scarves -0.22, Loungewear and Nightwear -0.13.

## 6. Error analysis

Fusion made 129 errors out of 1,500. Compared with text-only: fusion fixed 364, broke 38, both wrong 91. Six random examples (`misclassified_grid.png`):

| # | Title | True | Fusion | Text | Image | Likely cause |
|---|---|---|---|---|---|---|
| 1022 | Colorbar ... Lip Gloss 017 | Lips | Nails | Lips | Fragrance | Image ambiguous (small tube); "Lip Gloss" removed by masking. Fusion overrode the correct text. |
| 1218 | Nike Women Black Sling Bag | Bags | Shoes | Bags | Bags | Both single models were right, fusion wrong. "Nike" titles in train are mostly Shoes/Topwear, only 28 of 342 are Bags (hypothesis: brand prior). |
| 843 | iPanema Women Ritmos Black Heels | Shoes | Flip Flops | Shoes | Sandal | Label questionable: the image is a thong-strap sandal, and Shoes/Sandal/Flip Flops overlap. "iPanema" in train: 16 Flip Flops, 5 Shoes. |
| 287 | Hanes Men Grey ... Lounge Pants | Loungewear | Bottomwear | Innerwear | Bottomwear | Label questionable: the image shows plain trousers. |
| 992 | Sepia Women Printed Red Dress | Dress | Topwear | Topwear | Topwear | Image ambiguous at low resolution (dress looks like a long top); "Dress" removed from text. |
| 387 | Lakme Absolute Matte Merlot Lipstick 45 | Lips | Nails | Nails | Fragrance | Text uninformative after masking ("Lakme" is split across Makeup 12, Nails 10, Lips 10). The image is clear to a human, so the image model is weak here. |

## 7. Verdict: did fusion help?

**Yes.** Macro F1: +0.047 over image-only (0.7373 vs 0.6905) and +0.198 over text-only (0.7373 vs 0.5394). Accuracy: +0.044 and +0.217. Against text-only, fusion fixed 364 samples and broke 38. Here the image was the stronger single modality because class words were masked from titles.

**Limits:**
1. Single run and split, no confidence intervals; some classes have only 3 test samples.
2. Fusion has an MLP head with a hidden layer, models A and B have linear heads, so part of the gain may come from extra capacity (not tested).
3. Fusion is LR-sensitive (0.5042 at LR 1e-2).
4. Fusion can hurt (38 samples, e.g. #1218).
5. The conclusion applies to masked titles only.

**Not done:** raw-title experiment, multiple seeds, encoder fine-tuning, class weighting, 224x224 images, all bonus items, per-class fusion vs image-only comparison.
