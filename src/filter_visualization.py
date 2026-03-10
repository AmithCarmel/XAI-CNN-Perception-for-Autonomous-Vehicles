"""
filter_visualization.py
-----------------------
Gradient Ascent Filter Visualization for VGG16.

Usage in notebooks:
    from tensorflow.keras.applications import VGG16
    model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))

    # NOT build_vgg16() — that wraps the base and hides the conv layers

Why VGG16 base directly:
    build_vgg16() nests VGG16 inside a single 'vgg16' layer.
    VGG16(include_top=False) exposes block1_conv1, block2_conv1 etc. at top level.

Why VGG16 over EfficientNetB0 for filter visualization:
    VGG16 has simple Conv2D -> ReLU structure — gradients flow cleanly.
    EfficientNetB0 uses depthwise separable convs + batch norm which
    interfere with gradient ascent and produce flat/noisy results.

Author : Amith Carmel Anthony Raj
Project: Explainable CNN Perception for Autonomous Vehicles
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from typing import List, Optional, Tuple
import os
import sys

PROJECT_ROOT = "C:/CNN_Perception_AV"
if PROJECT_ROOT + "/src" not in sys.path:
    sys.path.insert(0, PROJECT_ROOT + "/src")

# VGG16 works in [0, 1] pixel space
MODEL_INPUT_SIZE = 224
PIXEL_MIN        = 0.0
PIXEL_MAX        = 1.0
PIXEL_INIT_LOW   = 0.4
PIXEL_INIT_HIGH  = 0.6


def build_feature_extractor(model: tf.keras.Model,
                             layer_name: str) -> tf.keras.Model:
    """
    Sub-model that outputs activations at a specific layer.

    Requires model to have conv layers accessible at the top level.
    Use VGG16(include_top=False) not build_vgg16().
    """
    return tf.keras.Model(
        inputs  = model.inputs,
        outputs = model.get_layer(layer_name).output,
        name    = f"extractor_{layer_name}"
    )


def filter_loss(feature_extractor: tf.keras.Model,
                img_tensor: tf.Variable,
                filter_index: int,
                use_freq_penalty: bool = True,
                freq_weight: float = 1e-4) -> tf.Tensor:
    """
    Loss = mean activation of target filter
         - (optional) TV penalty for smoothness
    """
    activation = feature_extractor(img_tensor)
    loss       = tf.reduce_mean(activation[:, :, :, filter_index])

    if use_freq_penalty:
        tv   = tf.reduce_sum(tf.image.total_variation(img_tensor))
        loss = loss - freq_weight * tf.cast(tv, tf.float32)

    return loss


def gradient_ascent_step(feature_extractor: tf.keras.Model,
                          img_tensor: tf.Variable,
                          filter_index: int,
                          learning_rate: float = 10.0,
                          use_freq_penalty: bool = True) -> Tuple[tf.Variable, float]:
    """One gradient ascent step."""
    with tf.GradientTape() as tape:
        loss = filter_loss(feature_extractor, img_tensor,
                           filter_index, use_freq_penalty)

    grads = tape.gradient(loss, img_tensor)
    grads = grads / (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
    img_tensor.assign_add(learning_rate * grads)

    return img_tensor, float(loss)


def visualize_filter(model: tf.keras.Model,
                     layer_name: str,
                     filter_index: int,
                     img_size: int = 128,
                     iterations: int = 30,
                     learning_rate: float = 10.0,
                     use_freq_penalty: bool = True,
                     num_restarts: int = 3) -> np.ndarray:
    """
    Generate an image that maximally activates a specific VGG16 filter.

    Parameters
    ----------
    model        : VGG16(include_top=False) — NOT build_vgg16()
    layer_name   : any VGG16 conv layer name e.g. 'block3_conv1'
    filter_index : which filter channel to maximise
    img_size     : output display size (any value — generation always at 224)
    iterations   : gradient ascent steps (30 is good for VGG16)
    learning_rate: step size (10.0 works well)
    use_freq_penalty: TV regularization for smoother output
    num_restarts : returns best result from N random starts

    Good VGG16 layers to try:
      block1_conv1 → simple edges, colour blobs
      block2_conv1 → corners, simple textures
      block3_conv1 → complex textures
      block4_conv1 → object parts
      block5_conv3 → high-level semantic patterns
    """
    feature_extractor = build_feature_extractor(model, layer_name)
    best_loss = -np.inf
    best_img  = None

    for restart in range(num_restarts):

        img_data   = np.random.uniform(
            PIXEL_INIT_LOW, PIXEL_INIT_HIGH,
            (1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3)
        ).astype(np.float32)

        img_tensor = tf.Variable(img_data)

        for step in range(iterations):
            img_tensor, loss = gradient_ascent_step(
                feature_extractor, img_tensor,
                filter_index, learning_rate, use_freq_penalty
            )
            img_tensor.assign(tf.clip_by_value(img_tensor, PIXEL_MIN, PIXEL_MAX))

        if loss > best_loss:
            best_loss = loss
            best_img  = img_tensor[0].numpy()

    img_uint8 = np.uint8(best_img * 255)

    if img_size != MODEL_INPUT_SIZE:
        img_uint8 = cv2.resize(img_uint8, (img_size, img_size),
                               interpolation=cv2.INTER_LANCZOS4)
    return img_uint8


def visualize_layer_filters(model: tf.keras.Model,
                             layer_name: str,
                             n_filters: int = 16,
                             img_size: int = 128,
                             iterations: int = 30,
                             cols: int = 8,
                             save_path: Optional[str] = None) -> np.ndarray:
    """Visualize first N filters of a layer as a grid."""
    rows   = (n_filters + cols - 1) // cols
    margin = 4
    cell   = img_size + margin
    grid   = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)

    for i in range(n_filters):
        print(f"  Filter {i+1}/{n_filters} in '{layer_name}'...")
        img = visualize_filter(model, layer_name,
                               filter_index=i,
                               img_size=img_size,
                               iterations=iterations)
        row, col = divmod(i, cols)
        y0, x0   = row * cell, col * cell
        grid[y0:y0+img_size, x0:x0+img_size] = img

    fig, ax = plt.subplots(figsize=(cols * 2, rows * 2))
    ax.imshow(grid)
    ax.set_title(
        f"Filter Visualizations — {layer_name}  ({n_filters} filters)",
        fontsize=13, fontweight="bold"
    )
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[OK] Saved to {save_path}")

    plt.show()
    return grid


def compare_layers(model: tf.keras.Model,
                   layer_names: List[str],
                   filter_index: int = 0,
                   img_size: int = 128,
                   iterations: int = 30,
                   save_path: Optional[str] = None):
    """
    Visualize same filter across multiple layers — shallow to deep.

    layer_names should be ordered from shallow to deep e.g.:
    ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    """
    n         = len(layer_names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.suptitle(
        f"Filter #{filter_index} — shallow to deep",
        fontsize=13, fontweight="bold"
    )

    for ax, layer_name in zip(axes, layer_names):
        try:
            img = visualize_filter(model, layer_name,
                                   filter_index=filter_index,
                                   img_size=img_size,
                                   iterations=iterations)
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{e}",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="red")
        ax.set_title(layer_name, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[OK] Saved to {save_path}")
    plt.show()


def compare_tv_regularization(model: tf.keras.Model,
                               layer_name: str,
                               filter_index: int = 0,
                               img_size: int = 128,
                               iterations: int = 30,
                               save_path: Optional[str] = None):
    """
    Side-by-side comparison of filter visualization with/without TV regularization.

    Left  : Without TV — noisy checkerboard artifacts
    Right : With TV    — smoother, more interpretable pattern
    """
    print("  Generating without TV regularization...")
    img_no_tv = visualize_filter(
        model, layer_name, filter_index,
        img_size=img_size, iterations=iterations,
        use_freq_penalty=False, num_restarts=1
    )

    print("  Generating with TV regularization...")
    img_tv = visualize_filter(
        model, layer_name, filter_index,
        img_size=img_size, iterations=iterations,
        use_freq_penalty=True, num_restarts=1
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(
        f"TV Regularization — {layer_name}, Filter {filter_index}",
        fontsize=13, fontweight="bold"
    )
    axes[0].imshow(img_no_tv)
    axes[0].set_title("Without TV Regularization\n(noise/checkerboard artifacts)")
    axes[0].axis("off")

    axes[1].imshow(img_tv)
    axes[1].set_title("With TV Regularization\n(smoother, cleaner pattern)")
    axes[1].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[OK] Saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    from tensorflow.keras.applications import VGG16

    print("Loading VGG16 base for filter visualization...")
    model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    print("Testing block1_conv1 filter 0...")
    img = visualize_filter(model, "block1_conv1", filter_index=0,
                           img_size=128, iterations=30, num_restarts=1)
    print(f"[OK] Output shape: {img.shape}")
