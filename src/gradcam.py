"""
gradcam.py
----------
Grad-CAM and Guided Grad-CAM implementation using TensorFlow GradientTape.

Reference: Selvaraju et al., 2017 — Grad-CAM: Visual Explanations from
           Deep Networks via Gradient-based Localization

Author : Amith Carmel Anthony Raj
Project: Explainable CNN Perception for Autonomous Vehicles
"""

import os
import sys

# ── SET YOUR PROJECT PATH HERE ────────────────────────────────────────────────
PROJECT_ROOT = "C:/CNN_Perception_AV"

if PROJECT_ROOT + "/src" not in sys.path:
    sys.path.insert(0, PROJECT_ROOT + "/src")
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from typing import Optional, Tuple


# ── Core Grad-CAM ─────────────────────────────────────────────────────────────

class GradCAM:
    """
    Computes Grad-CAM heatmaps for a given model and target layer.

    Usage
    -----
    gcam     = GradCAM(model, last_conv_layer_name="block5_conv3")
    heatmap  = gcam.compute(img_array, class_index=1)
    overlay  = gcam.overlay(original_img, heatmap)
    """

    def __init__(self, model: tf.keras.Model, last_conv_layer_name: str):
        self.model               = model
        self.last_conv_layer_name = last_conv_layer_name

        self.grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(last_conv_layer_name).output,
                model.output
            ]
        )

    def compute(self,
                img_array: np.ndarray,
                class_index: Optional[int] = None) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.

        Parameters
        ----------
        img_array   : preprocessed image (1, H, W, 3)
        class_index : target class; if None uses argmax of predictions

        Returns
        -------
        heatmap : 2D numpy array, values in [0, 1]
        """
        img_tensor = tf.cast(img_array, tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_tensor)
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            class_score = predictions[:, class_index]

        grads       = tape.gradient(class_score, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()

    @staticmethod
    def overlay(original_img: np.ndarray,
                heatmap: np.ndarray,
                alpha: float = 0.4,
                colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """Superimpose Grad-CAM heatmap onto the original image."""
        h, w = original_img.shape[:2]

        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8   = np.uint8(255 * heatmap_resized)
        heatmap_color   = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_color   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        overlay = (heatmap_color * alpha + original_img * (1 - alpha)).astype(np.uint8)
        return overlay

    def get_predicted_class(self, img_array: np.ndarray,
                            class_names: list) -> Tuple[str, float]:
        """Return predicted class name and confidence."""
        preds = self.model.predict(img_array, verbose=0)
        idx   = np.argmax(preds[0])
        return class_names[idx], float(preds[0][idx])


# ── Guided Grad-CAM ───────────────────────────────────────────────────────────

class GuidedGradCAM:
    """
    Guided Grad-CAM = element-wise product of Guided Backprop x Grad-CAM.
    Produces sharper, class-discriminative saliency maps.
    """

    def __init__(self, model: tf.keras.Model, last_conv_layer_name: str):
        self.model   = model
        self.gradcam = GradCAM(model, last_conv_layer_name)
        self._build_guided_model()

    def _build_guided_model(self):
        """Replace ReLU with GuidedReLU for backprop masking."""

        @tf.custom_gradient
        def guided_relu(x):
            def grad(dy):
                return dy * tf.cast(dy > 0, dtype=tf.float32) * \
                             tf.cast(x  > 0, dtype=tf.float32)
            return tf.nn.relu(x), grad

        model_config  = self.model.get_config()
        guided_model  = tf.keras.Model.from_config(model_config)
        guided_model.set_weights(self.model.get_weights())

        for layer in guided_model.layers:
            if hasattr(layer, "activation"):
                if layer.activation == tf.keras.activations.relu:
                    layer.activation = guided_relu

        self.guided_model = guided_model

    def compute(self,
                img_array: np.ndarray,
                class_index: Optional[int] = None) -> np.ndarray:
        """
        Compute Guided Grad-CAM saliency map.

        Returns
        -------
        guided_gradcam : (H, W, 3) float array
        """
        heatmap = self.gradcam.compute(img_array, class_index)
        h, w    = img_array.shape[1:3]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        img_tensor = tf.cast(img_array, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            preds = self.guided_model(img_tensor)
            if class_index is None:
                class_index = tf.argmax(preds[0])
            class_score = preds[:, class_index]

        guided_grads = tape.gradient(class_score, img_tensor)[0].numpy()
        guided_grads = np.maximum(guided_grads, 0)
        guided_grads /= (guided_grads.max() + 1e-8)

        return guided_grads * heatmap_resized[..., np.newaxis]


# ── Visualization helpers ─────────────────────────────────────────────────────

def plot_gradcam_comparison(original_img: np.ndarray,
                            heatmap: np.ndarray,
                            overlay_img: np.ndarray,
                            class_name: str,
                            confidence: float,
                            save_path: Optional[str] = None):
    """Plot original | heatmap | overlay side-by-side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Grad-CAM  |  Prediction: {class_name}  ({confidence*100:.1f}%)",
        fontsize=14, fontweight="bold"
    )

    axes[0].imshow(original_img);             axes[0].set_title("Original Image");    axes[0].axis("off")
    im = axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Grad-CAM Heatmap"); axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].imshow(overlay_img);              axes[2].set_title("Overlay");           axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[OK] Saved to {save_path}")
    plt.show()


def plot_multi_class_gradcam(model: tf.keras.Model,
                              img_array: np.ndarray,
                              original_img: np.ndarray,
                              class_names: list,
                              last_conv_layer: str,
                              save_path: Optional[str] = None):
    """Generate Grad-CAM for every class in a grid."""
    gcam = GradCAM(model, last_conv_layer)
    n    = len(class_names)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))
    fig.suptitle("Grad-CAM per Class", fontsize=16, fontweight="bold")

    for i, cname in enumerate(class_names):
        heatmap = gcam.compute(img_array, class_index=i)
        overlay = GradCAM.overlay(original_img, heatmap)

        axes[0, i].imshow(heatmap, cmap="jet"); axes[0, i].set_title(f"Heatmap — {cname}"); axes[0, i].axis("off")
        axes[1, i].imshow(overlay);             axes[1, i].set_title(f"Overlay — {cname}"); axes[1, i].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[OK] Saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    print("[INFO] gradcam.py loaded. Import GradCAM or GuidedGradCAM to use.")
