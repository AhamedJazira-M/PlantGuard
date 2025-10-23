import tensorflow as tf
import os

# -------------------------------
# Paths and settings
# -------------------------------
RAW_DATA_DIR = "data/raw"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15  # total epochs you want to run
SEED = 123

# -------------------------------
# Load datasets
# -------------------------------
train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    RAW_DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    RAW_DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# -------------------------------
# Class names
# -------------------------------
class_names = train_ds_raw.class_names
NUM_CLASSES = len(class_names)
print(f"Classes found: {class_names}")

# -------------------------------
# Normalize images
# -------------------------------
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds_raw.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds_raw.map(lambda x, y: (normalization_layer(x), y))

# -------------------------------
# Data augmentation
# -------------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1)
])
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# -------------------------------
# Prefetch for performance
# -------------------------------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# -------------------------------
# Check if checkpoint exists
# -------------------------------
checkpoint_path = os.path.join(MODEL_DIR, "plantguard_best_model.h5")
if os.path.exists(checkpoint_path):
    print(f"✅ Loading checkpoint from {checkpoint_path}")
    model = tf.keras.models.load_model(checkpoint_path)
else:
    print("No checkpoint found, building new model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

# -------------------------------
# Compile model
# -------------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# Checkpoint callback
# -------------------------------
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# -------------------------------
# Train the model
# -------------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb]
)

# -------------------------------
# Save final model
# -------------------------------
final_model_path = os.path.join(MODEL_DIR, "plantguard_final_model.h5")
model.save(final_model_path)
print(f"✅ Final model saved to {final_model_path}")
