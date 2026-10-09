import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # run from anywhere; paths below are relative to this folder
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report
import json, time

from data_loader import load_fashion_mnist

np.random.seed(42)
tf.random.set_seed(42)
sns.set_style("whitegrid")

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# ============================================================
# 1. LOAD DATA
# ============================================================
(X_train_full, y_train_full), (X_test, y_test) = load_fashion_mnist()
print("Full train:", X_train_full.shape, "Test:", X_test.shape)

# Hold out a validation set from training data (stratified by simple slicing since balanced)
val_frac = 0.1
n_val = int(len(X_train_full) * val_frac)
rng = np.random.RandomState(42)
idx = rng.permutation(len(X_train_full))
val_idx, train_idx = idx[:n_val], idx[n_val:]
X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]
print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# ============================================================
# 2. EDA — sample images + class distribution
# ============================================================
fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
for i, ax in enumerate(axes.flat):
    idx_c = np.where(y_train == i)[0][0]
    ax.imshow(X_train[idx_c], cmap="gray")
    ax.set_title(CLASS_NAMES[i], fontsize=10)
    ax.axis("off")
plt.suptitle("Fashion-MNIST — One Example per Class", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/fig1_sample_images.png", dpi=140)
plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
counts = np.bincount(y_train_full)
ax.bar(CLASS_NAMES, counts, color="#4C72B0")
ax.set_title("Class Distribution — Full Training Set (60,000 images)")
ax.set_xlabel("Class")
ax.set_ylabel("Number of Images")
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig("figures/fig2_class_distribution.png", dpi=140)
plt.close()

# ============================================================
# 3. PREPROCESSING
# ============================================================
def preprocess(X):
    X = X.astype("float32") / 255.0
    return X[..., np.newaxis]  # add channel dimension -> (N, 28, 28, 1)

X_train_p = preprocess(X_train)
X_val_p = preprocess(X_val)
X_test_p = preprocess(X_test)

print("Preprocessed shape:", X_train_p.shape, "dtype:", X_train_p.dtype, "range:", X_train_p.min(), X_train_p.max())

# ============================================================
# 4. DATA AUGMENTATION LAYER (applied only during training)
# ============================================================
data_augmentation = keras.Sequential([
    layers.RandomTranslation(0.08, 0.08),
    layers.RandomZoom(0.08),
    layers.RandomFlip("horizontal"),  # clothing items are left-right symmetric-ish; safe for this domain
], name="data_augmentation")

# ============================================================
# 5. MODEL ARCHITECTURE — CNN
# ============================================================
def build_cnn():
    inputs = keras.Input(shape=(28, 28, 1), name="input_image")
    x = data_augmentation(inputs)

    x = layers.Conv2D(32, 3, padding="same", activation="relu", name="conv1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, name="pool1")(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="conv3")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="conv4")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, name="pool2")(x)
    x = layers.Dropout(0.25)(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(10, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="fashion_cnn")
    return model

model = build_cnn()
model.summary()

# Save architecture summary to text file for the report
with open("results/model_summary.txt", "w") as f:
    model.summary(print_fn=lambda line: f.write(line + "\n"))

total_params = model.count_params()
trainable_params = sum(np.prod(v.shape) for v in model.trainable_weights)
print(f"\nTotal params: {total_params:,} | Trainable: {trainable_params:,}")

# ============================================================
# 6. COMPILE & TRAIN
# ============================================================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
]

start = time.time()
history = model.fit(
    X_train_p, y_train,
    validation_data=(X_val_p, y_val),
    epochs=12,
    batch_size=128,
    callbacks=callbacks,
    verbose=2,
)
train_time = time.time() - start
print(f"\nTraining time: {train_time:.1f}s, stopped at epoch {len(history.history['loss'])}")

with open("results/history.json", "w") as f:
    json.dump(history.history, f)

# ============================================================
# 7. TRAINING CURVES
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
axes[0].plot(history.history["loss"], label="Training Loss", color="#4C72B0")
axes[0].plot(history.history["val_loss"], label="Validation Loss", color="#C44E52")
axes[0].set_title("Loss Curve")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(history.history["accuracy"], label="Training Accuracy", color="#4C72B0")
axes[1].plot(history.history["val_accuracy"], label="Validation Accuracy", color="#C44E52")
axes[1].set_title("Accuracy Curve")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].legend()
plt.tight_layout()
plt.savefig("figures/fig3_training_curves.png", dpi=140)
plt.close()

# ============================================================
# 8. TEST SET EVALUATION
# ============================================================
test_loss, test_acc = model.evaluate(X_test_p, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

y_proba = model.predict(X_test_p, verbose=0)
y_pred = np.argmax(y_proba, axis=1)

report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
report_txt = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
print("\n", report_txt)
with open("results/classification_report.txt", "w") as f:
    f.write(report_txt)

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar_kws={"label": "Count"})
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title(f"Confusion Matrix — Test Set (Accuracy = {test_acc:.4f})")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("figures/fig4_confusion_matrix.png", dpi=140)
plt.close()

# Per-class accuracy bar chart
per_class_acc = cm.diagonal() / cm.sum(axis=1)
fig, ax = plt.subplots(figsize=(8, 4.2))
bars = ax.bar(CLASS_NAMES, per_class_acc, color="#55A868")
ax.axhline(test_acc, color="#C44E52", linestyle="--", label=f"Overall accuracy ({test_acc:.3f})")
ax.set_title("Per-Class Recall (Test Set)")
ax.set_ylabel("Recall")
ax.tick_params(axis="x", rotation=30)
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("figures/fig5_per_class_accuracy.png", dpi=140)
plt.close()

# ============================================================
# 9. MISCLASSIFIED EXAMPLES
# ============================================================
wrong_idx = np.where(y_pred != y_test)[0]
rng2 = np.random.RandomState(1)
sample_wrong = rng2.choice(wrong_idx, size=10, replace=False)

fig, axes = plt.subplots(2, 5, figsize=(11, 5))
for ax, i in zip(axes.flat, sample_wrong):
    ax.imshow(X_test[i], cmap="gray")
    ax.set_title(f"True: {CLASS_NAMES[y_test[i]]}\nPred: {CLASS_NAMES[y_pred[i]]}", fontsize=9)
    ax.axis("off")
plt.suptitle("Sample Misclassified Test Images", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/fig6_misclassified.png", dpi=140)
plt.close()

print(f"\nTotal misclassified: {len(wrong_idx)} / {len(y_test)} ({len(wrong_idx)/len(y_test)*100:.2f}%)")

# ============================================================
# 10. ABLATION: baseline dense-only model (no CNN, no augmentation, no dropout)
#     to quantify how much the CNN architecture + regularisation actually help
# ============================================================
def build_baseline_mlp():
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Flatten()(inputs)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="baseline_mlp")

# Use a training subset for the ablation study only, to keep runtime reasonable;
# still large enough (10,000 images) to give a fair, meaningful comparison.
X_train_sub = X_train_p[:10000]
y_train_sub = y_train[:10000]

baseline = build_baseline_mlp()
baseline.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
baseline_history = baseline.fit(
    X_train_sub, y_train_sub, validation_data=(X_val_p, y_val),
    epochs=8, batch_size=128, verbose=2,
)
baseline_test_loss, baseline_test_acc = baseline.evaluate(X_test_p, y_test, verbose=0)
print(f"\nBaseline MLP test accuracy: {baseline_test_acc:.4f} (vs CNN {test_acc:.4f})")

# CNN without augmentation/dropout for a cleaner ablation of "does regularisation help"
def build_cnn_no_reg():
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="cnn_no_reg")

cnn_no_reg = build_cnn_no_reg()
cnn_no_reg.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
cnn_no_reg_history = cnn_no_reg.fit(
    X_train_sub, y_train_sub, validation_data=(X_val_p, y_val),
    epochs=8, batch_size=128, verbose=2,
)
cnn_no_reg_test_loss, cnn_no_reg_test_acc = cnn_no_reg.evaluate(X_test_p, y_test, verbose=0)
print(f"CNN (no augmentation/dropout, 8 epochs, 10k subset) test accuracy: {cnn_no_reg_test_acc:.4f}")

ablation = {
    "Baseline MLP (10k subset, 8 epochs)": float(baseline_test_acc),
    "CNN, no augmentation/dropout (10k subset, 8 epochs)": float(cnn_no_reg_test_acc),
    "Final CNN (full 54k train, augmentation+BatchNorm+Dropout, 12 epochs)": float(test_acc),
}
with open("results/ablation_results.json", "w") as f:
    json.dump(ablation, f, indent=2)
print("\nAblation summary:", ablation)

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.bar(list(ablation.keys()), list(ablation.values()), color=["#C44E52", "#DD8452", "#55A868"])
ax.set_ylabel("Test Accuracy")
ax.set_title("Ablation Study — Impact of Architecture and Regularisation")
ax.set_ylim(0, 1)
ax.tick_params(axis="x", rotation=15)
for i, v in enumerate(ablation.values()):
    ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
plt.tight_layout()
plt.savefig("figures/fig7_ablation.png", dpi=140)
plt.close()

print("\nAll figures, model summary, history, and reports saved.")
print(f"FINAL TEST ACCURACY: {test_acc:.4f}")
print(f"FINAL TEST LOSS: {test_loss:.4f}")
print(f"TOTAL PARAMS: {total_params:,}")
print(f"EPOCHS RUN: {len(history.history['loss'])}")
print(f"TRAINING TIME: {train_time:.1f}s")
