import tensorflow as tf
import os

# Paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    RAW_DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    RAW_DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# Get class names before mapping
class_names = train_ds.class_names

# Normalize
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Save class labels
label_map_path = os.path.join(PROCESSED_DATA_DIR, "class_indices.txt")
with open(label_map_path, "w") as f:
    for idx, class_name in enumerate(class_names):
        f.write(f"{idx}: {class_name}\n")

print("✅ Preprocessing complete!")
print(f"Total training batches: {tf.data.experimental.cardinality(train_ds).numpy()}")
print(f"Total validation batches: {tf.data.experimental.cardinality(val_ds).numpy()}")
print(f"Classes found: {class_names}")
