# Chest X-Ray Pneumonia Detector

Classifies chest X-ray images as **NORMAL** or **PNEUMONIA** using a
convolutional neural network built with transfer learning.

This is the second project in a healthcare-AI portfolio series, moving from
tabular sensor data (see the
[activity-classifier-java](../activity-classifier-java) project) into
**image classification and deep learning**.

## Why transfer learning

Training a CNN from scratch needs a huge amount of data and compute to learn
basic visual features (edges, textures, shapes) before it can even start
learning anything specific to X-rays. Instead, this project starts from
**MobileNetV2**, a model already trained on ImageNet (millions of general
photos), and fine-tunes it on X-ray images. The early layers already know
how to detect general visual patterns — training only has to teach the model
what's specific to distinguishing a healthy lung from a pneumonia-affected
one. This is standard practice in real-world medical imaging work, not just
a shortcut for this project.

## Dataset

[Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia),
originally from Kermany et al., 2018. ~5,863 X-ray images across three
pre-split folders: `train`, `val`, `test`, each containing `NORMAL/` and
`PNEUMONIA/` subfolders.

**You'll need a free Kaggle account** to download it (Kaggle requires
authentication, unlike the UCI dataset in the last project):

1. Create a Kaggle account, go to Account → Create New API Token — this
   downloads a `kaggle.json` file.
2. Place that file at `~/.kaggle/kaggle.json` (Mac/Linux) or
   `C:\Users\<you>\.kaggle\kaggle.json` (Windows).
3. Run `python src/download_data.py`.

## Project structure

```
pneumonia-detector/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                    # dataset lands here (gitignored, ~1.2GB)
├── results/                 # training plots + evaluation report (generated)
└── src/
    ├── download_data.py     # downloads dataset via Kaggle API
    ├── train_model.py       # builds, trains, and evaluates the CNN
    └── predict.py           # runs the trained model on a single image
```

## Setup and usage

```bash
git clone <your-repo-url>
cd pneumonia-detector
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

python src/download_data.py   # needs kaggle.json set up first
python src/train_model.py     # trains and evaluates, saves results
python src/predict.py path/to/some_xray.jpg   # try it on one image
```

## What the training script does

1. Loads images from `data/chest_xray/{train,val,test}` using Keras'
   `image_dataset_from_directory`, resizing everything to 224×224 (what
   MobileNetV2 expects) and batching them.
2. Applies light data augmentation to the training set (random flips,
   rotation, zoom) — this helps the model generalize instead of memorizing
   exact pixel patterns, which matters a lot given the dataset isn't huge.
3. Loads MobileNetV2 pretrained on ImageNet, **freezes its weights** (so
   training doesn't wreck what it already learned), and adds a small new
   classification head on top (pooling → dense → dropout → sigmoid output).
4. Trains only the new head for several epochs.
5. **Fine-tunes**: unfreezes the top portion of MobileNetV2 and trains a bit
   further at a much lower learning rate — this lets the model adjust its
   higher-level visual features specifically for X-rays, without destroying
   the general features learned from ImageNet.
6. Evaluates on the held-out test set: accuracy, precision, recall, F1,
   confusion matrix, and ROC-AUC. Saves training curves and the confusion
   matrix as images in `results/`.

## Results

*(Fill in after running it.)*

| Metric | Value |
|---|---|
| Test accuracy | — |
| Precision | — |
| Recall | — |
| F1 score | — |
| ROC-AUC | — |

## Why recall matters more than accuracy here

In a medical screening context, **false negatives are worse than false
positives** — missing a real pneumonia case is more dangerous than flagging
a healthy X-ray for a second look. Because of this, recall on the
PNEUMONIA class deserves more attention than overall accuracy alone. The
evaluation script reports both, but this README will call out the tradeoff
once real numbers are in.

## Limitations

- Trained on a single public dataset from one source (pediatric patients at
  one hospital system in China) — accuracy would likely drop on X-rays from
  different populations, equipment, or imaging protocols.
- The dataset's class balance is skewed toward PNEUMONIA in training data,
  which can bias a naively-trained model — the training script accounts for
  this with class weighting, but it's a real limitation worth knowing about.
- This is a learning project, not a validated diagnostic tool, and should
  never be treated as one. Real clinical deployment would require far more
  rigorous validation, regulatory approval, and clinician oversight.

## Next steps

- Try a different backbone (EfficientNet, ResNet) and compare.
- Add Grad-CAM visualization to show *which parts* of an X-ray the model is
  focusing on — useful both for debugging and for explainability.
- Multi-class version: bacterial vs. viral pneumonia vs. normal, instead of
  binary classification.
