# Emotion-Support Learning Platform: Facial Expression Recognition for Enhanced Student Engagement
Welcome to the Emotion Recognition for Online Learning Student Support repository. This project is developed as part of my master’s dissertation and aims to harness the power of emotion recognition to provide real-time emotional support to students during their learning journey on an educational platform.

The core idea behind this project is to enhance the student experience by identifying their emotional states and delivering tailored feedback and encouragement based on these emotions. This initiative seeks to not only improve academic performance but also promote emotional well-being in online learning environments.

<p align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png" style="height:50px">
    <img src="https://miro.medium.com/v2/resize:fit:512/1*IMGOKBIN8qkOBt5CH55NSw.png" style="height:50px;">
    <img src="https://user-images.githubusercontent.com/59691442/172961027-fd9185a5-da77-46e3-97b1-54e99e242822.png" alt="opencvLogo" style="height:50px;">
    <img src="https://www.fullstackpython.com/img/logos/react.png" alt="kerasLogo" style="height:50px;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/d/d5/CSS3_logo_and_wordmark.svg" alt="kerasLogo" style="height:50px;">
</p>

## Description
Facial Expression Recognition (FER) in python with PyTorch and OpenCV.

✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨

[👉🏽👉🏽👉🏽Try the FER sytem directly in your Browser !!👈🏽👈🏽👈🏽]()

✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨

## Expression recognition
The Platform is capable of detecting facial expressions on Real-Time. The possible emotions to be detected are as follows: 

- Neutral
- Sadness
- Happiness
- Fear
- Anger
- Surprise
- Disgust
- Contempt (Only for the AffectNet models)


## Background
Different CNN architectures including custom CNN, VGGNet, ResNet, and EfficientNet were used to develop the Facial Expression Recognition (FER) Models which were integrated into a custom online learning platform for this project to develop and emotionally aware learning platform. 

### Features
The FER System features :
- Learning Platform: A web-based educational platform designed to help students engage with their coursework, enhanced by the addition of emotional intelligence that supports their learning process.
- Real-Time Emotion Detection: Utilizing a Convolutional Neural Network (CNN) trained on an extensive dataset, the platform can detect emotions such as anger, contempt, disgust, fear, happiness, neutrality, sadness, and surprise from students’ facial expressions.
- Personalized Emotional Support: Based on the detected emotions, the platform provides relevant motivational messages and support, helping students manage stress and stay engaged during their studies.
- Tutors Feedback and Insights: Tutors are provided with real-time insights into their students’ emotional states, helping them better understand how students are engaging with the material. This data can be used to tailor teaching approaches, offer additional support, and improve overall student outcomes.
- Admin Dashboard: The platform features an intuitive admin dashboard where administrators can monitor overall system performance, manage user data, and gain insights into the emotional trends of students across different learning sessions. This dashboard offers a comprehensive overview of both individual and group-level emotional data.
- Web-Based Interface: The platform is designed with a user-friendly interface that allows seamless integration of emotion recognition into virtual classrooms or self-paced learning environments.
- Privacy-Focused: The system only processes live video feed data temporarily for emotion detection and does not store any personal video data, ensuring student privacy.

### Technology Stack
	•	Python (Flask, OpenCV, PyTorch)
	•	Machine Learning: CNN-based emotion recognition model
	•	Computer Vision: Face detection using Haar cascades and image preprocessing
	•	Frontend: HTML5, CSS3, JavaScript
	•	Deployment: Flask web application for serving the platform

## Datasets
Two datasets (FER2013 and AffectNet processed version) were used for training/validation. The datasets consist of different emotional expressions with FER2013 and AffectNet processed having 7 and 8 emotions respectively.

Note: Due to the space limitations for upload on Github, it waas not possible to upload the dataset we used.

### Requirements
- Python 3.9+
- TorchVision
- OpenCV
- [PyTorch (with CUDA for GPU usage)](https://pytorch.org/get-started/locally/)
- All other requirements listed in [**requirements.txt**](requirements.txt)

### Training
The training process involves optimising the different CNN models to accuartely understand and identify different facial emotions.Various hyperparameters such as batch size, learning rate, optimiser, and loss function in the `config.yaml` file. Please refer to the `config.yaml` file for a full list of hyperparameters.

### File Structure
- `train.py`: File to initiate the training loop for the respective model, which uses the hyperparameters in the `config.yaml` file. This script initializes the training loop using the hyperparameters specified in config.yaml. It is designed to work with various models.
- `config.yaml`: File to edit hyperparameters. Additional hyperparameters are included for choosing project and logger names for 'wandb'.
- `models.py`: Contains the definitions of the seq2seq/transformer models.
- `dataset.py`: Creates our datasets in the relevant PyTorch format.
- `utils.py`: Utility functions for calculating BLEU, NIST, etc.
- `logger.py`: Logger class for logging training metrics to 'wandb'.
- `checkpoints/`:  A directory where model checkpoints are saved during training. These checkpoints can be loaded later for resuming training or performing inference.

Note: Because of the file size for checkpoints, it was not possible to upload on Github.

### Usage
##### 1. Download/Clone the Repository
First, download or clone the entire repository. To avoid issues, we recommend following the folder structure as found in this repository. If you modify the structure, you may need to update file paths for datasets and checkpoints accordingly.

##### 2. Install Dependencies
Ensure that all libraries listed in the requirements.txt file are installed. We suggest creating a fresh environment to avoid dependency conflicts. You can install the required libraries using the command below:
        ```terminal
              pip install -r requirements.txt
         ```
Additionally, you will need to install PyTorch. For GPU support, install the CUDA version by following the instructions on the PyTorch website.

##### 3. Download Datasets
You’ll need a training and validation dataset (or a single dataset split into both). For our experiments, we used the ‘FER2013 and AffectNet processed’ dataset, with an 80%/20% split between training and validation. A placeholder folder is provided in the repository, but you can download the datasets here. This download also includes test datasets for inference.

Steps:

1. Delete the placeholder dataset folder in the repository.
2. Download and extract the dataset into the same location, ensuring the folder structure matches the original repository layout.

##### 4. Run Training
After setting up the datasets and installing dependencies, you can run the training script:
        ```terminal
           pip train.py
          ```
Before training, you may want to edit the config.yaml file to customize hyperparameters (e.g., learning rate, batch size) or specify the checkpoint paths for saving your models.

Upon installing all packages and training the model. The app could be strated as follows: 

First cd into emotions-recognition directory. The front end will start on http://localhost:3000
N.B. run the front end on `app.py` by typing in the terminal the following commands :

```bash
start
```

and 

for the backend, cd into backends, it starts on http://localhost:5000

```bash
python app.py
```

##### 5.Use Checkpoints
You can download our pre-trained model checkpoints from our checkpoints. Once downloaded, place them in the /checkpoints directory of the relevant model folder. Ensure you update the config.yaml file to include the path to the desired checkpoint.

##### 6.Folder Structure Considerations
The code expects that you open your workspace in one of the following directories: /Models for the model training and architecture files, /backend for the system backend, emotion-recongnition for the frontend. This setup ensures the correct relative paths are used during training. You can modify the structure as needed, but we recommend sticking to the default setup for simplicity.

### Inference
To run model inference in the backend, replace the .pth file referenced in app.py (located in the /backend folder) with one of your desired .pth files. Ensure that the pre-trained checkpoints are downloaded and placed in the checkpoints/ folder.

You can modify the file paths to point to your chosen model checkpoint directly within the app.py code. If you have your own trained models, you can also use those by adjusting the path accordingly.


<p align="center">
    <img src="https://github.com/user-attachments/assets/0cf00623-d6ad-44a2-9675-7dba078df0fd" alt="quentinVid"/>
    <img src="https://github.com/user-attachments/assets/032f50ec-f7ba-4f99-a8b0-a85130e19a6d" alt="clementVid"/>
    <img src="https://github.com/user-attachments/assets/8503a2f4-d3d0-4dee-94de-654071f95018" alt="yohanVid"/>
</p>

### Weights & Biases ('wandb')
If you have not used 'wandb' previously, you will be prompted to enter your API key into the terminal. You need a (free) 'wandb' account if not already made, and you can find further instructions [here](https://docs.wandb.ai/quickstart).


## Libraries

- [Python](https://www.python.org)
- [OpenCV](https://opencv.org)
- [OpenCV weights](https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml)
- [Flask](https://flask.palletsprojects.com/en/2.1.x/)
- [PyTorch](https://pytorch.org/)
- [React](https://react.dev/)

## Author 
Abdulmalik Abdullahi Shinkafi

### Research Contribution
This project contributes to the growing field of affective computing and EdTech by exploring the integration of artificial intelligence into online learning platforms. It offers insights into how emotion recognition can be applied to improve student engagement, emotional well-being, and overall learning outcomes. By providing tutors with emotional insights and supporting students with personalized feedback, the platform promotes a more empathetic and effective learning environment.
