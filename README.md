# Alzheimer-disease-detection
Overview:
This project consists of two main components: Manually Extracting Gait Features and Attention-based ADualNet for dual-task classification. Below is a detailed description of each part:

1. Manually Extract Gait Features
In this part, we utilize MMPose, a powerful open-source toolbox, to detect human keypoints from video frames. These keypoints represent the positions of various body joints during motion. Based on these detected keypoints, we manually extract 15 gait features that describe the characteristics of human walking. These features are critical for tasks that involve human movement analysis, such as gait recognition or medical evaluations.

2. Attention-based ADualNet for Dual-task Classification
We construct an attention-based ADualNet model for dual-task classification using both speech and video inputs. The dataset used for this task originates from a medical application in which subjects perform a dual-task. The following components are used to build the model:

  wav2vec: Extracts deep speech features from the audio data.
  I3D: Extracts deep video features from the visual data.

The extracted features from both modalities are processed and aligned, followed by a self-attention and cross-attention mechanism to model the interdependencies between the modalities. The final network performs a binary classification task with an accuracy of 0.62, evaluated using a 5-fold cross-validation approach.
