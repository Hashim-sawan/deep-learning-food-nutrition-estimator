# Food Image Nutrition Estimator

Predicts calories, mass, protein, carbohydrates, and fat from a single overhead photo of a dish. Built as my individual project for APS360 (Applied Fundamentals of Deep Learning) at the University of Toronto, 2026.

The task is a multi-task regression, not classification. The model has to account for both what the food is and how much of it is on the plate, so it outputs five continuous values instead of a food label.

## Results

Test split of Nutrition5k (n = 496), held out until all models were fixed. MAE as a percentage of the target mean in parentheses.

| Target | Mean predictor | kNN (k = 5) | ResNet-50 | EfficientNet-B0 |
|---|---|---|---|---|
| Calories (kcal) | 160.5 (65.6%) | 92.1 (37.7%) | **56.6 (23.1%)** | 58.6 (24.0%) |
| Mass (g) | 113.5 (54.1%) | 60.8 (29.0%) | **34.8 (16.6%)** | 36.8 (17.6%) |
| Protein (g) | 14.3 (82.3%) | 8.3 (47.9%) | **5.7 (32.9%)** | 5.8 (33.6%) |
| Carbohydrates (g) | 13.2 (71.5%) | 8.7 (46.9%) | 6.1 (32.8%) | **5.9 (32.1%)** |
| Fat (g) | 10.3 (83.3%) | 6.5 (52.5%) | **4.2 (34.0%)** | 4.5 (36.6%) |

Calorie metrics on the same split:

| Model | MAE (kcal) | RMSE (kcal) | R² | Acc@20% |
|---|---|---|---|---|
| Mean predictor | 160.5 | 198.3 | 0.00 | 18.3% |
| kNN (k = 5) | 92.1 | 129.3 | 0.57 | 33.1% |
| ResNet-50 | 56.6 | 82.7 | 0.83 | 44.4% |
| EfficientNet-B0 | 58.6 | 85.6 | 0.81 | 46.2% |

Acc@20% is the share of dishes predicted within ±20% of true calories.

For reference, the Nutrition5k paper reports 70.6 kcal (26.1%) for its RGB-only direct prediction model on the full test set. This project's test set excludes dishes above the 99th percentile in calories or mass, so the numbers are not directly comparable.

Main findings:

- Fine-tuned ResNet-50 cuts calorie MAE from 160.5 to 56.6 kcal (a 65% reduction over the mean predictor) and beats kNN on every target.
- EfficientNet-B0 comes within 4% of ResNet-50 on calorie MAE with 5.3x fewer parameters, but degrades much more on out-of-distribution images (see below).
- The kNN baseline on frozen ImageNet features already cuts calorie error by 43%. Much of the task is recognizing the food, and fine-tuning adds the rest.
- Mass is the easiest target in relative terms, and the macronutrients are the hardest, since protein and fat depend on composition that a photo does not show.

## Data

[Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) (Thames et al., 2021), released by Google Research: real cafeteria dishes with per-dish calories, mass, and macronutrients. The dataset is not included in this repo. The notebook downloads the metadata CSVs, official split files, and overhead RGB images from the public Google Cloud Storage bucket.

Processing pipeline:

1. Custom parser for the metadata CSVs. Rows are variable in length, which breaks pandas parsing, so the parser takes the first six fields per row and skips corrupted rows (5,006 dishes).
2. Drop dishes without an overhead `rgb.png` (3,490 remaining).
3. Remove outliers above the 99th percentile in calories (908.7 kcal) or mass (696.1 g), and dishes with non-positive mass (3,427 remaining).
4. Apply the official train/test split and carve a 15% validation set from training with a fixed seed: 2,265 train / 438 validation / 496 test.
5. Z-score each target using training-set statistics only. Predictions are converted back to original units for evaluation.
6. Resize to 256 and crop to 224 x 224. Training uses random crops, horizontal flips, and colour jitter. ImageNet normalization throughout.

## Model

- **Backbone:** ResNet-50 with ImageNet weights, classification head removed, giving a 2048-d pooled feature vector.
- **Head:** shared dense block (2048 → 512, ReLU, dropout 0.3) branching into five independent linear heads (512 → 1), one per target. 24.6M parameters total.
- **Training:** mean Huber loss on z-scored targets, Adam (lr 1e-4, batch size 64), 15 epochs. The backbone is frozen for the first 3 epochs so the new heads warm up before fine-tuning. The checkpoint with the lowest validation loss is kept. One run takes about 13 minutes on a Colab T4 GPU.
- **Comparison model:** the same head on an EfficientNet-B0 backbone (4.7M parameters).

Baselines:

- **Mean predictor:** outputs the training-set mean of each target for every dish. This is the floor any image-based model has to beat.
- **kNN (k = 5):** encodes each image with a frozen ImageNet ResNet-50 and averages the targets of the 5 nearest training images. This tests how far pretrained features go without gradient training.

## Out-of-distribution test

I photographed 11 prepackaged dishes from a gym cafeteria in Saudi Arabia, overhead, with ground truth from the printed nutrition labels. One item was dropped because its label macronutrients did not match its calories (n = 10). The lids stayed on, which makes this set deliberately harder than Nutrition5k. It was never used for training or tuning.

| Model | Calories (kcal) | Protein (g) | Carbs (g) | Fat (g) |
|---|---|---|---|---|
| ResNet-50 | 154.6 | 13.3 | 24.0 | 6.4 |
| EfficientNet-B0 | 242.5 | 29.3 | 26.4 | 3.9 |

ResNet-50's calorie MAE here is close to what the mean predictor gets on the test split, so most of the learned advantage disappears off-distribution. Familiar-looking dishes are predicted almost exactly (broccoli chicken caesar: 348 true, 348 predicted), while dense dishes are underestimated by about half (turkey alfredo pasta: 595 true, 211 predicted). Lid glare, a dark reflective table, ceiling lighting, and dishes unlike those in Nutrition5k all contribute. The smaller EfficientNet-B0 is less robust.

## Limitations

- The model regresses toward the mean at high calories (a 902 kcal salad bowl predicted at 400) and can misread composition (a rice bowl predicted at 46 g protein against a true 7 g).
- Validation MAE (45.8 kcal) is lower than test MAE (56.6 kcal), likely because near-duplicate captures of the same plate land on both sides of the random train/validation split. The official test split avoids this overlap.
- Hidden ingredients like oils and sauces are almost invisible in a photo and get underestimated.
- Error is likely uneven across cuisines that are underrepresented in Nutrition5k, as the self-collected results suggest.
- A 57 kcal average error per dish compounds over a day of meals. This model should not be used for medical nutrition management such as insulin dosing.
- The planned depth-map variant was dropped because training it was not feasible on the free Colab tier. It is the natural next step for the hidden-volume errors.

## Running it

1. Open `nutrition_estimator.ipynb` in Google Colab with a GPU runtime (T4 is enough).
2. Run the data cells to download Nutrition5k and cache the images to Google Drive, so later sessions restore them without redownloading.
3. Run the remaining cells top to bottom to train the baselines and both models and reproduce the tables above.

Dependencies are listed in `requirements.txt`.

## References

- Q. Thames et al. Nutrition5k: Towards automatic nutritional understanding of generic food. CVPR, 2021.
- K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. CVPR, 2016.
- M. Tan and Q. V. Le. EfficientNet: Rethinking model scaling for convolutional neural networks. ICML, 2019.
