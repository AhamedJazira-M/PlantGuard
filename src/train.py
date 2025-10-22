# src/train.py
import tensorflow as tf
import os

# ==============================
# PATHS AND SETTINGS
# ==============================
RAW_DATA_DIR = "data/raw"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 123

# ==============================
# LOAD DATASETS
# ==============================
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

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print(f"✅ Classes found: {class_names}")

# ==============================
# DATA AUGMENTATION + NORMALIZATION
# ==============================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1)
])

# Normalize + prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = (
    train_ds
    .map(lambda x, y: (data_augmentation(x, training=True), y))
    .map(lambda x, y: (tf.keras.layers.Rescaling(1./255)(x), y))
    .cache()
    .shuffle(1000)
    .prefetch(buffer_size=AUTOTUNE)
)
val_ds = (
    val_ds
    .map(lambda x, y: (tf.keras.layers.Rescaling(1./255)(x), y))
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

# ==============================
# BUILD MODEL
# ==============================
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

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# CALLBACKS (SAVE BEST MODEL)
# ==============================
checkpoint_path = os.path.join(MODEL_DIR, "plantguard_best_model.h5")
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    monitor="val_accuracy",
    mode="max",
    verbose=1
)

# ==============================
# TRAIN MODEL
# ==============================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

# ==============================
# SAVE FINAL MODEL
# ==============================
final_model_path = os.path.join(MODEL_DIR, "plantguard_final_model.h5")
model.save(final_model_path)
print(f"✅ Training complete! Models saved to:\n- Best: {checkpoint_path}\n- Final: {final_model_path}")
