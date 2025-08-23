import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import wandb
from wandb.keras import WandbCallback

# ---------------- Data ----------------
def load_fer2013_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    # FER2013 has columns: emotion, pixels, Usage
    pixels = df["pixels"].str.split().apply(lambda p: np.array(p, dtype=np.uint8))
    X = np.stack(pixels.values)  # (N, 2304)
    X = X.reshape((-1, 48, 48, 1))  # grayscale
    y = df["emotion"].astype(int).values
    usage = df["Usage"].values
    # normalize to [0,1]
    X = X.astype("float32") / 255.0
    return X, y, usage

def split_by_usage(X, y, usage):
    train_idx = usage == "Training"
    val_idx = usage == "PrivateTest"
    test_idx = usage == "PublicTest"
    return (X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])

# --------------- Model ----------------
def build_cnn(num_classes: int):
    inputs = keras.Input(shape=(48,48,1))
    x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model

# --------------- Training -------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True, help="Path to fer2013.csv")
    parser.add_argument("--project", default="ml-end-to-end")
    parser.add_argument("--run_name", default="cnn-fer2013")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # W&B init
    wandb.init(project=args.project, name=args.run_name, config=vars(args))

    X, y, usage = load_fer2013_csv(args.csv_path)
    X_train, y_train, X_val, y_val, X_test, y_test = split_by_usage(X, y, usage)

    num_classes = int(np.max(y) + 1)
    model = build_cnn(num_classes)
    opt = keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        WandbCallback(save_model=False),
        keras.callbacks.ModelCheckpoint("model.keras", save_best_only=True, monitor="val_accuracy", mode="max")
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})
    print(f"Test accuracy: {test_acc:.4f}")

    # Save labeled classes
    np.save("classes.npy", np.unique(y))

    # Save final model
    model.save("model.keras")

    # Log as artifact
    at = wandb.Artifact("emotion-cnn", type="model")
    at.add_file("model.keras")
    at.add_file("classes.npy")
    wandb.log_artifact(at)

    wandb.finish()

if __name__ == "__main__":
    main()
