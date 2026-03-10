"""
saliency.py
-----------
Vanilla Saliency Maps, SmoothGrad, and Integrated Gradients.

Author : Amith Carmel Anthony Raj
Project: Explainable CNN Perception for Autonomous Vehicles
"""

import os
import sys

# ── PROJECT PATH ────────────────────────────────────────────────
PROJECT_ROOT = "C:/CNN_Perception_AV"

if PROJECT_ROOT + "/src" not in sys.path:
    sys.path.insert(0, PROJECT_ROOT + "/src")
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Optional


# ── 1. Vanilla Saliency ───────────────────────────────────────────────────────

class VanillaSaliency:
    """
    Gradient of class score w.r.t. input pixels.
    Highlights pixels that most affect the prediction if perturbed.
    """

    def __init__(self, model: tf.keras.Model):
        self.model = model

    def compute(self,
                img_array: np.ndarray,
                class_index: Optional[int] = None) -> np.ndarray:
        """
        Parameters
        ----------
        img_array   : (1, H, W, 3) preprocessed image
        class_index : target class; defaults to argmax prediction

        Returns
        -------
        saliency : (H, W) numpy array, absolute gradient magnitudes
        """
        img_tensor = tf.Variable(tf.cast(img_array, tf.float32))

        with tf.GradientTape() as tape:
            preds = self.model(img_tensor)
            if class_index is None:
                class_index = tf.argmax(preds[0]).numpy()
            class_score = preds[:, class_index]

        grads    = tape.gradient(class_score, img_tensor)[0].numpy()
        saliency = np.max(np.abs(grads), axis=-1)
        saliency /= (saliency.max() + 1e-8)
        return saliency


# ── 2. SmoothGrad ─────────────────────────────────────────────────────────────

class SmoothGrad:
    """
    Reduces noise in saliency maps by averaging gradients over
    N copies of the image with added Gaussian noise.

    Reference: Smilkov et al., 2017
    """

    def __init__(self, model: tf.keras.Model,
                 num_samples: int = 50,
                 noise_std: float = 0.15):
        self.model      = model
        self.num_samples = num_samples
        self.noise_std  = noise_std
        self._vanilla   = VanillaSaliency(model)

    def compute(self,
                img_array: np.ndarray,
                class_index: Optional[int] = None) -> np.ndarray:
        """
        Returns
        -------
        smooth_saliency : (H, W) averaged saliency map
        """
        total_grads = np.zeros(img_array.shape[1:3])

        for _ in range(self.num_samples):
            noise  = np.random.normal(0, self.noise_std, img_array.shape).astype(np.float32)
            noisy  = img_array + noise
            sal    = self._vanilla.compute(noisy, class_index)
            total_grads += sal

        smooth  = total_grads / self.num_samples
        smooth /= (smooth.max() + 1e-8)
        return smooth


# ── 3. Integrated Gradients ───────────────────────────────────────────────────

class IntegratedGradients:
    """
    Integrates gradients along a path from baseline to input.

    Reference: Sundararajan et al., 2017 — Axiomatic Attribution for Deep Networks
    """

    def __init__(self, model: tf.keras.Model, steps: int = 50):
        self.model = model
        self.steps = steps

    def compute(self,
                img_array: np.ndarray,
                class_index: Optional[int] = None,
                baseline: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Parameters
        ----------
        img_array   : (1, H, W, 3) preprocessed image
        class_index : target class; defaults to argmax prediction
        baseline    : reference image (default: all-zeros / black image)

        Returns
        -------
        integrated_grads : (H, W, 3) attribution map
        """
        if baseline is None:
            baseline = np.zeros_like(img_array)

        if class_index is None:
            preds       = self.model.predict(img_array, verbose=0)
            class_index = np.argmax(preds[0])

        alphas       = np.linspace(0, 1, self.steps)
        interpolated = np.array([
            baseline + alpha * (img_array - baseline)
            for alpha in alphas
        ], dtype=np.float32).squeeze(axis=1)

        grads    = self._batch_gradients(interpolated, class_index)
        avg_grads = np.mean(grads, axis=0)
        return (img_array[0] - baseline[0]) * avg_grads

    def _batch_gradients(self, interpolated: np.ndarray,
                         class_index: int) -> np.ndarray:
        all_grads  = []
        batch_size = 10

        for i in range(0, len(interpolated), batch_size):
            batch = tf.Variable(tf.cast(interpolated[i:i+batch_size], tf.float32))
            with tf.GradientTape() as tape:
                preds = self.model(batch)
                score = preds[:, class_index]
            grads = tape.gradient(score, batch).numpy()
            all_grads.append(grads)

        return np.concatenate(all_grads, axis=0)

    @staticmethod
    def to_heatmap(integrated_grads: np.ndarray) -> np.ndarray:
        """Convert (H, W, 3) attribution map to (H, W) scalar heatmap."""
        heatmap  = np.sum(np.abs(integrated_grads), axis=-1)
        heatmap /= (heatmap.max() + 1e-8)
        return heatmap


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_saliency_comparison(original_img: np.ndarray,
                              vanilla_sal: np.ndarray,
                              smooth_sal: np.ndarray,
                              ig_sal: np.ndarray,
                              class_name: str,
                              save_path: Optional[str] = None):
    """Side-by-side: Original | Vanilla | SmoothGrad | Integrated Gradients"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Saliency Methods  |  Class: {class_name}",
                 fontsize=14, fontweight="bold")

    imgs   = [original_img, vanilla_sal, smooth_sal, ig_sal]
    titles = ["Original", "Vanilla Saliency", "SmoothGrad", "Integrated Gradients"]
    cmaps  = [None, "hot", "hot", "hot"]

    for ax, img, title, cmap in zip(axes, imgs, titles, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[OK] Saved to {save_path}")
    plt.show()


def overlay_saliency(original_img: np.ndarray,
                     saliency: np.ndarray,
                     percentile: int = 95,
                     color: tuple = (255, 0, 0)) -> np.ndarray:
    """Overlay top-percentile saliency pixels on the original image."""
    threshold   = np.percentile(saliency, percentile)
    mask        = saliency >= threshold
    highlighted = original_img.copy()

    for c, val in enumerate(color):
        channel           = highlighted[:, :, c].astype(np.float32)
        channel[mask]     = 0.5 * channel[mask] + 0.5 * val
        highlighted[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

    return highlighted


if __name__ == "__main__":
    print("[INFO] saliency.py loaded. Available: VanillaSaliency, SmoothGrad, IntegratedGradients")
