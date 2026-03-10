"""
active_safety_analysis.py
--------------------------
Grad-CAM based failure analysis for autonomous vehicle perception.

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
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, List, Dict
from gradcam import GradCAM


# ── Active Safety Scenario Definitions ───────────────────────────────────────

SAFETY_SCENARIOS = {
    "blind_spot_detection": {
        "description"  : "Detect vehicles in the rear-lateral blind zone",
        "critical_classes": ["car", "cyclist"],
        "miss_penalty" : "HIGH — risk of side collision during lane change"
    },
    "safe_door_open": {
        "description"  : "Detect cyclists/pedestrians before door opens",
        "critical_classes": ["pedestrian", "cyclist"],
        "miss_penalty" : "HIGH — risk of dooring cyclist"
    },
    "lane_change_assist": {
        "description"  : "Detect approaching vehicles in target lane",
        "critical_classes": ["car"],
        "miss_penalty" : "HIGH — risk of lane change collision"
    },
    "pedestrian_crossing": {
        "description"  : "Detect pedestrians at intersections",
        "critical_classes": ["pedestrian"],
        "miss_penalty" : "CRITICAL — pedestrian safety"
    }
}


# ── Failure Case Analyzer ─────────────────────────────────────────────────────

class ActiveSafetyAnalyzer:
    """
    Wraps Grad-CAM to produce scenario-specific failure reports.

    For each test image:
      1. Run inference — check if critical class was detected
      2. Generate Grad-CAM for predicted class
      3. Generate Grad-CAM for expected class
      4. Compare — if model focused on wrong region, that is a failure
    """

    def __init__(self, model: tf.keras.Model,
                 class_names: List[str],
                 last_conv_layer: str):
        self.model       = model
        self.class_names = class_names
        self.gcam        = GradCAM(model, last_conv_layer)

    def analyze(self,
                img_array: np.ndarray,
                original_img: np.ndarray,
                scenario: str,
                expected_class: str) -> Dict:
        """
        Run analysis for a single image under a given safety scenario.

        Parameters
        ----------
        img_array      : (1, H, W, 3) preprocessed
        original_img   : (H, W, 3) uint8 for visualization
        scenario       : key from SAFETY_SCENARIOS
        expected_class : the class that should be the top prediction

        Returns
        -------
        report dict with prediction, confidence, failure flag, heatmaps
        """
        preds           = self.model.predict(img_array, verbose=0)[0]
        predicted_idx   = int(np.argmax(preds))
        predicted_class = self.class_names[predicted_idx]
        confidence      = float(preds[predicted_idx])
        expected_idx    = self.class_names.index(expected_class)
        expected_conf   = float(preds[expected_idx])
        is_failure      = (predicted_class != expected_class)

        heatmap_predicted = self.gcam.compute(img_array, class_index=predicted_idx)
        overlay_predicted = GradCAM.overlay(original_img, heatmap_predicted)
        heatmap_expected  = self.gcam.compute(img_array, class_index=expected_idx)
        overlay_expected  = GradCAM.overlay(original_img, heatmap_expected)

        return {
            "scenario"             : scenario,
            "scenario_info"        : SAFETY_SCENARIOS.get(scenario, {}),
            "predicted_class"      : predicted_class,
            "predicted_confidence" : confidence,
            "expected_class"       : expected_class,
            "expected_confidence"  : expected_conf,
            "is_failure"           : is_failure,
            "failure_severity"     : SAFETY_SCENARIOS.get(scenario, {}).get("miss_penalty", "UNKNOWN"),
            "heatmap_predicted"    : heatmap_predicted,
            "heatmap_expected"     : heatmap_expected,
            "overlay_predicted"    : overlay_predicted,
            "overlay_expected"     : overlay_expected,
            "all_class_probs"      : {
                self.class_names[i]: float(preds[i])
                for i in range(len(self.class_names))
            }
        }

    def plot_analysis(self, report: Dict,
                      original_img: np.ndarray,
                      save_path: Optional[str] = None):
        """Visualise the analysis report as a 2x3 panel."""
        fig          = plt.figure(figsize=(18, 10))
        fig.patch.set_facecolor("#1a1a2e")
        status_color = "#e74c3c" if report["is_failure"] else "#2ecc71"
        status_text  = "FAILURE" if report["is_failure"] else "PASS"

        fig.suptitle(
            f"Active Safety Analysis — {report['scenario'].replace('_', ' ').title()}  |  {status_text}",
            fontsize=14, fontweight="bold", color=status_color, y=0.98
        )

        ax1 = fig.add_subplot(2, 3, 1)
        ax1.imshow(original_img)
        ax1.set_title("Input Image", color="white")
        ax1.axis("off")

        ax2 = fig.add_subplot(2, 3, 2)
        ax2.imshow(report["heatmap_predicted"], cmap="jet")
        ax2.set_title(
            f"Predicted: {report['predicted_class']} ({report['predicted_confidence']*100:.1f}%)",
            color="#f39c12" if report["is_failure"] else "#2ecc71"
        )
        ax2.axis("off")

        ax3 = fig.add_subplot(2, 3, 3)
        ax3.imshow(report["overlay_predicted"])
        ax3.set_title("Predicted Overlay", color="white")
        ax3.axis("off")

        ax4 = fig.add_subplot(2, 3, 4)
        classes    = list(report["all_class_probs"].keys())
        probs      = list(report["all_class_probs"].values())
        bar_colors = [
            "#e74c3c" if c == report["predicted_class"] else
            "#2ecc71" if c == report["expected_class"]  else
            "#3498db"
            for c in classes
        ]
        bars = ax4.barh(classes, probs, color=bar_colors)
        ax4.set_xlim(0, 1)
        ax4.set_title("Class Probabilities", color="white")
        ax4.tick_params(colors="white")
        ax4.set_facecolor("#16213e")
        for spine in ax4.spines.values():
            spine.set_edgecolor("#444")
        for bar, prob in zip(bars, probs):
            ax4.text(prob + 0.01, bar.get_y() + bar.get_height()/2,
                     f"{prob:.2f}", va="center", color="white", fontsize=9)

        legend_patches = [
            mpatches.Patch(color="#e74c3c", label=f"Predicted: {report['predicted_class']}"),
            mpatches.Patch(color="#2ecc71", label=f"Expected: {report['expected_class']}"),
        ]
        ax4.legend(handles=legend_patches, loc="lower right",
                   facecolor="#1a1a2e", labelcolor="white", fontsize=8)

        ax5 = fig.add_subplot(2, 3, 5)
        ax5.imshow(report["heatmap_expected"], cmap="jet")
        ax5.set_title(
            f"Expected: {report['expected_class']} ({report['expected_confidence']*100:.1f}%)",
            color="#2ecc71"
        )
        ax5.axis("off")

        ax6 = fig.add_subplot(2, 3, 6)
        ax6.imshow(report["overlay_expected"])
        ax6.set_title("Expected Class Overlay", color="white")
        ax6.axis("off")

        if report["is_failure"]:
            fig.text(0.5, 0.01,
                     f"Failure Severity: {report.get('failure_severity', '')}",
                     ha="center", fontsize=10, color="#e74c3c", style="italic")

        for ax in [ax1, ax2, ax3, ax5, ax6]:
            ax.set_facecolor("#16213e")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"[OK] Saved to {save_path}")
        plt.show()

    def batch_report(self,
                     test_cases: List[Dict],
                     save_dir: Optional[str] = None) -> List[Dict]:
        """
        Run analysis on a list of test cases and print a summary table.

        Each test_case dict must have:
          img_array, original_img, scenario, expected_class, name
        """
        reports = []
        for tc in test_cases:
            report              = self.analyze(
                tc["img_array"], tc["original_img"],
                tc["scenario"],  tc["expected_class"]
            )
            report["test_name"] = tc.get("name", "unnamed")
            reports.append(report)

            if save_dir:
                path = os.path.join(save_dir, f"{tc.get('name', 'test')}_analysis.png")
                self.plot_analysis(report, tc["original_img"], save_path=path)

        print("\n" + "=" * 70)
        print(f"{'Test Name':<25} {'Expected':<15} {'Predicted':<15} {'Status'}")
        print("=" * 70)
        for r in reports:
            status = "FAIL" if r["is_failure"] else "PASS"
            print(f"{r['test_name']:<25} {r['expected_class']:<15} "
                  f"{r['predicted_class']:<15} {status}")
        print("=" * 70)
        n_fail = sum(r["is_failure"] for r in reports)
        print(f"\nTotal: {len(reports)} | Passed: {len(reports)-n_fail} | Failed: {n_fail}")

        return reports


if __name__ == "__main__":
    print("[INFO] active_safety_analysis.py loaded.")
    print("[INFO] Available scenarios:", list(SAFETY_SCENARIOS.keys()))
