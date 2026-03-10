Explainable CNN Perception for Autonomous Vehicles

Author: Amith Carmel Anthony Raj
MS ECE, University of Pittsburgh | AirLab CMU Collaborator | ADAS Intern, Veoneer
Built on top of: Visualizing_Filters_of_a_CNN_using_TensorFlow by Tim Sainburg


OVERVIEW

This project extends CNN filter visualization with production-grade explainability
tools applied directly to autonomous vehicle perception scenarios. The three core
additions over the companion repo are Grad-CAM heatmaps, saliency attribution maps,
and a structured ADAS failure analysis framework that classifies prediction errors
by safety severity.

The central research question is: when a CNN perception model makes the wrong call
in a safety-critical driving scene, can we explain exactly where it looked and why
it failed — using only the model's own gradients, with no extra labels or supervision?
The results show that we can, consistently, across all four tested ADAS scenarios.


WHAT THIS PROJECT ADDS

The companion repo covers gradient ascent filter visualization on a standard VGG16
network. This project keeps that as notebook 01 and extends it with the following:

- filter_visualization.py adds TV regularization to remove checkerboard artifacts,
  a multi-restart strategy that returns the best result across N random seeds, and
  a cross-layer comparison tool that shows how features evolve from shallow to deep.

- gradcam.py implements Grad-CAM and Guided Grad-CAM for any target layer, with
  per-class heatmap generation and overlay utilities.

- saliency.py implements three methods: Vanilla Saliency (fast gradient magnitude),
  SmoothGrad (averaged over 50 noisy copies, significantly cleaner), and Integrated
  Gradients (axiomatic attribution along a 50-step baseline-to-input path).

- active_safety_analysis.py wraps all of the above into a structured ADAS failure
  analyzer that takes a scenario type, an expected class, runs Grad-CAM for both the
  predicted and expected class, and flags the result with a severity label.

- model.py builds EfficientNetB0 with a partial unfreezing strategy, data
  augmentation pipeline, and training callbacks. VGG16 and ResNet50 builders are
  included for compatibility.


ADAS SCENARIOS AND FINDINGS

Notebook 04 runs the failure analyzer on four real driving images, one per scenario.

Blind Spot Detection: the model predicted cyclist at 99.7% on an image where a large
truck occupied the left edge. The cyclist filled the center frame and dominated the
pixel count. The expected class was car (the truck), which received only 0.3%.
Grad-CAM confirmed activation on the cyclist body, not the vehicle.

Safe Door Open: the model predicted car at 79.5% on a street scene with a yellow
taxi and an adjacent cyclist. The taxi is large and center-frame. The cyclist — the
actual dooring risk — was predicted at only 20.5%. Grad-CAM activation covers the
taxi hood and roof, completely ignoring the cyclist lane.

Lane Change Assist: three cyclists rode in the foreground with an SUV behind them.
The model predicted cyclist at 93.4%. The SUV, which is the lane-change collision
risk, received only 6.5%. Grad-CAM confirmed the hotspot lands on the cyclists,
not the vehicle in the adjacent lane.

Pedestrian Crossing: a crosswalk scene with cyclists, pedestrians, and a car. The
model predicted cyclist at 53.7% and car at 46%. The pedestrian — carrying a
CRITICAL severity flag — received exactly 0% confidence despite being the
primary safety concern at an intersection.

The consistent pattern across all four cases is dominant-class bias: the model
attends to whatever object fills the most pixels, not whatever poses the highest
safety risk. This is not a random failure — it is a structural limitation of
single-label scene classification applied to multi-object ADAS scenarios, and
Grad-CAM makes that limitation visible and spatially verifiable.


PROJECT STRUCTURE

src/ contains all Python modules: model.py, filter_visualization.py, gradcam.py,
saliency.py, and active_safety_analysis.py.

notebooks/ contains five Jupyter notebooks numbered 00 through 04. Run them in
order. Notebook 00 handles dataset sorting and training and only needs to be run
once. Notebooks 01 through 04 can be re-run independently after that.

data/driving_scenes/ is the sorted image folder with four subfolders: car,
pedestrian, cyclist, and background.

results/ holds all generated output images and saved model weights. This folder
is gitignored and not pushed to the repository.


HOW TO RUN

Step 1 — clone the repo and install dependencies:
    git clone https://github.com/AmithCarmel/XAI-CNN-Perception-for-Autonomous-Vehicles.git
    cd XAI-CNN-Perception-for-Autonomous-Vehicles
    pip install -r requirements.txt

Step 2 — open any notebook and set the project path at the top of the first cell:
    PROJECT_ROOT = "C:/CNN_Perception_AV"

Step 3 — run notebook 00 once. Set KITTI_IMAGES to your dataset path. YOLOv8 will
auto-sort the images and EfficientNetB0 will train automatically. Weights are saved
to results/trained_weights.weights.h5.

Step 4 — run notebooks 01 through 04 in any order. Notebooks 02, 03, and 04 load
weights automatically if the file exists. Notebook 01 uses VGG16 directly and does
not require trained weights.

To use your own images in notebooks 02, 03, and 04, uncomment the load_from_file
line and pass your image path:
    img_array, original_img = load_from_file("C:/your/path/image.jpg")


DATASET

This project uses the KITTI Vision Benchmark Suite (Geiger et al., CVPR 2012),
specifically the 5,985 front-camera training images. The raw dataset has no
per-image class labels, so notebook 00 uses a pretrained YOLOv8 nano model to
detect objects in each image and assign a dominant class label automatically.

YOLO class IDs used: car maps to IDs 2, 3, 5, 7 (car, motorcycle, bus, truck).
Pedestrian maps to ID 0. Cyclist maps to ID 1. Images with no detections are
assigned to background.


MODEL

EfficientNetB0 is used for all prediction tasks. The base model is initialized
with ImageNet weights. Layers 0 through 99 are frozen to preserve low-level
features. Layers 100 onward are fine-tuned on the driving dataset. The head adds
GlobalAveragePooling, BatchNormalization, two dense layers with dropout, and a
4-class softmax output. Expected validation accuracy is 90 to 95 percent with
the full KITTI split.

VGG16 is used only in notebook 01 for filter visualization. Its simple
Conv2D-ReLU structure allows gradients to flow cleanly during ascent.
EfficientNetB0's depthwise separable convolutions and batch normalization layers
interfere with gradient ascent and produce flat or noisy outputs, making VGG16
the correct choice for this specific task.


REQUIREMENTS

tensorflow 2.11 or higher, numpy, opencv-python, matplotlib, Pillow, requests,
ultralytics (for YOLOv8 auto-sort in notebook 00).

Install with: pip install -r requirements.txt

Note for Windows users: TensorFlow 2.11 and above does not support GPU on native
Windows. Use WSL2 or the TensorFlow-DirectML plugin for GPU acceleration. The
project runs fully on CPU without any code changes.


ACKNOWLEDGEMENTS

Tim Sainburg — companion repo on CNN filter visualization.
KITTI Vision Benchmark Suite — Geiger et al., CVPR 2012.
Grad-CAM — Selvaraju et al., ICCV 2017.
Integrated Gradients — Sundararajan et al., ICML 2017.
SmoothGrad — Smilkov et al., 2017.


LICENSE

MIT License