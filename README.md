Explainable CNN Perception for Autonomous Vehicles

Author: Amith Carmel Anthony Raj
MS ECE, University of Pittsburgh | AirLab CMU Collaborator 

ABOUT

This project builds an explainability framework for CNN-based autonomous vehicle perception, extending gradient ascent filter visualization with Grad-CAM, SmoothGrad, and Integrated Gradients. EfficientNetB0 is trained on 5,985 KITTI driving images across four classes: car, pedestrian, cyclist, and background. The core contribution is a structured ADAS failure analyzer that runs across four safety scenarios, blind spot detection, safe door open, lane change assist, and pedestrian crossing, and uses Grad-CAM heatmaps to spatially explain where and why the model fails. The consistent finding is dominant-class bias: the model attends to the most visually prominent object in the scene rather than the most safety-critical one, confirmed across all four scenarios with HIGH or CRITICAL severity misses in three of the four cases.

<img width="2562" height="1489" alt="blind_spot_car_analysis" src="https://github.com/user-attachments/assets/0f854079-f59e-4c83-a4dd-6ae4f3c8cdc8" />
<img width="2552" height="1489" alt="door_open_cyclist_analysis" src="https://github.com/user-attachments/assets/e7933cb7-48ba-44b4-a957-221df140a57b" />
<img width="2552" height="1489" alt="lane_change_car_analysis" src="https://github.com/user-attachments/assets/93399ee4-044e-4e73-9c33-76afca7a1e30" />
<img width="2552" height="1489" alt="pedestrian_crossing_analysis" src="https://github.com/user-attachments/assets/e8f7a0ac-78cd-46e6-9447-75ddf18fe5d2" />

HOW TO RUN

Clone the repo and install dependencies:
    git clone https://github.com/AmithCarmel/XAI-CNN-Perception-for-Autonomous-Vehicles.git
    pip install -r requirements.txt

Open any notebook and set PROJECT_ROOT at the top of the first cell to your local project path. Run notebook 00 once to auto-sort your KITTI images using YOLOv8 and train the model — weights are saved automatically to results/trained_weights.weights.h5. After that, run notebooks 01 through 04 in any order. To use your own images in notebooks 02, 03, and 04, uncomment the load_from_file line and pass your image path.


LICENSE

MIT License - free to use, modify, and distribute with attribution.
