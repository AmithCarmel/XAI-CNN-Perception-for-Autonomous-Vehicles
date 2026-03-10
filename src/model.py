import os, sys, numpy as np, tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0, VGG16, ResNet50

PROJECT_ROOT = "C:/CNN_Perception_AV"
if PROJECT_ROOT + "/src" not in sys.path:
    sys.path.insert(0, PROJECT_ROOT + "/src")

CLASS_NAMES = ["background", "car", "pedestrian", "cyclist"]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE    = (224, 224)

def build_augmentation():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    ], name="augmentation")

def build_efficientnet(num_classes=NUM_CLASSES, freeze_base=False, use_augmentation=True):
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), name="input_image")
    x = inputs
    if use_augmentation:
        x = build_augmentation()(x)
    base = EfficientNetB0(weights="imagenet", include_top=False, input_tensor=x)
    base.trainable = not freeze_base
    if not freeze_base:
        for layer in base.layers[:100]:
            layer.trainable = False
        for layer in base.layers[100:]:
            layer.trainable = True
    x = base.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_top")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.4, name="drop1")(x)
    x = layers.Dense(128, activation="relu", name="fc2")(x)
    x = layers.Dropout(0.3, name="drop2")(x)
    output = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = Model(inputs=base.input, outputs=output, name="EfficientNetB0_AV")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_acc")]
    )
    return model

def build_vgg16(num_classes=NUM_CLASSES, freeze_base=False, use_augmentation=True):
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), name="input_image")
    x = inputs
    if use_augmentation:
        x = build_augmentation()(x)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    base = VGG16(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    base.trainable = not freeze_base
    x = base(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(512, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.5, name="drop1")(x)
    x = layers.Dense(256, activation="relu", name="fc2")(x)
    x = layers.Dropout(0.3, name="drop2")(x)
    output = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = Model(inputs=inputs, outputs=output, name="VGG16_AV")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_resnet50(num_classes=NUM_CLASSES, freeze_base=False, use_augmentation=True):
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), name="input_image")
    x = inputs
    if use_augmentation:
        x = build_augmentation()(x)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    base = ResNet50(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    base.trainable = not freeze_base
    x = base(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(512, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.4, name="drop1")(x)
    output = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = Model(inputs=inputs, outputs=output, name="ResNet50_AV")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def get_callbacks(save_dir=PROJECT_ROOT + "/results"):
    os.makedirs(save_dir, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=save_dir + "/best_model.keras",
            monitor="val_accuracy", save_best_only=True, verbose=1
        )
    ]

def build_dataset(data_dir, batch_size=32, val_split=0.2, seed=42):
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=val_split, subset="training",
        seed=seed, image_size=IMG_SIZE, batch_size=batch_size,
        label_mode="categorical"
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=val_split, subset="validation",
        seed=seed, image_size=IMG_SIZE, batch_size=batch_size,
        label_mode="categorical"
    )
    class_names = train_ds.class_names
    print(f"[INFO] Classes found      : {class_names}")
    print(f"[INFO] Training batches   : {len(train_ds)}")
    print(f"[INFO] Validation batches : {len(val_ds)}")
    return train_ds.prefetch(AUTOTUNE), val_ds.prefetch(AUTOTUNE), class_names

def preprocess_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    return np.expand_dims(arr, axis=0)

def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, (layers.Conv2D, layers.DepthwiseConv2D)):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")

def load_model_weights(model, weights_path):
    model.load_weights(weights_path)
    print(f"[OK] Loaded weights from {weights_path}")
    return model
