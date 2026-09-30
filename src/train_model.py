"""
Trains a pneumonia classifier from chest X-ray images using transfer
learning on MobileNetV2.

Run download_data.py first.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "chest_xray"
RESULTS_DIR = BASE_DIR / "results"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
HEAD_EPOCHS = 5
FINE_TUNE_EPOCHS = 5
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "train", image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="binary"
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "val", image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="binary"
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "test", image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="binary", shuffle=False,
    )
    return train_ds, val_ds, test_ds


def build_model():
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ])

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base_model.trainable = False  # freeze for initial head training

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


def get_class_weights(train_ds):
    labels = np.concatenate([y.numpy() for _, y in train_ds], axis=0).flatten()
    weights = compute_class_weight(
        class_weight="balanced", classes=np.array([0, 1]), y=labels
    )
    return {0: weights[0], 1: weights[1]}


def plot_training_curves(history_head, history_fine_tune):
    acc = history_head.history["accuracy"] + history_fine_tune.history["accuracy"]
    val_acc = history_head.history["val_accuracy"] + history_fine_tune.history["val_accuracy"]
    loss = history_head.history["loss"] + history_fine_tune.history["loss"]
    val_loss = history_head.history["val_loss"] + history_fine_tune.history["val_loss"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(acc, label="train")
    axes[0].plot(val_acc, label="validation")
    axes[0].axvline(HEAD_EPOCHS - 0.5, color="gray", linestyle="--", label="fine-tuning starts")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(loss, label="train")
    axes[1].plot(val_loss, label="validation")
    axes[1].axvline(HEAD_EPOCHS - 0.5, color="gray", linestyle="--", label="fine-tuning starts")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_curves.png", dpi=150)
    print(f"Saved training curves to {RESULTS_DIR / 'training_curves.png'}")


def evaluate(model, test_ds):
    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0).flatten()
    y_prob = model.predict(test_ds).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    auc = roc_auc_score(y_true, y_prob)
    print(report)
    print(f"ROC-AUC: {auc:.3f}")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    print(f"Saved confusion matrix to {RESULTS_DIR / 'confusion_matrix.png'}")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curve.png", dpi=150)
    print(f"Saved ROC curve to {RESULTS_DIR / 'roc_curve.png'}")

    with open(RESULTS_DIR / "results.txt", "w") as f:
        f.write("Pneumonia Detector — Results Summary\n")
        f.write("=====================================\n\n")
        f.write(report)
        f.write(f"\nROC-AUC: {auc:.3f}\n")
        f.write(f"\nConfusion matrix:\n{cm}\n")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading datasets...")
    train_ds, val_ds, test_ds = load_datasets()
    class_weights = get_class_weights(train_ds)
    print(f"Class weights (to offset dataset imbalance): {class_weights}")

    print("\nBuilding model (MobileNetV2 base, frozen)...")
    model, base_model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    print(f"\nTraining classification head for {HEAD_EPOCHS} epochs...")
    history_head = model.fit(
        train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS,
        class_weight=class_weights,
    )

    print("\nUnfreezing top layers of MobileNetV2 for fine-tuning...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False  # keep most of the base frozen, only fine-tune the top

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # much lower LR
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print(f"Fine-tuning for {FINE_TUNE_EPOCHS} more epochs...")
    history_fine_tune = model.fit(
        train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weights,
    )

    plot_training_curves(history_head, history_fine_tune)

    print("\nEvaluating on test set...")
    evaluate(model, test_ds)

    model.save(BASE_DIR / "pneumonia_model.keras")
    print(f"\nSaved trained model to {BASE_DIR / 'pneumonia_model.keras'}")


if __name__ == "__main__":
    main()
