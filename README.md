# Real and Fake Face Detection

A machine learning project to detect and classify real faces versus fake/manipulated faces using the [Real and Fake Face Detection dataset](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection) from Kaggle.

## Dataset

This project uses the "Real and Fake Face Detection" dataset from Kaggle, which contains:
- **Real Images**: Authentic facial photographs
- **Fake Images**: Manipulated facial images, including printed paper photos, digital screen images, and other synthetic faces

The dataset provides a balanced collection of real and fake facial images, making it suitable for training binary classification models to identify deepfakes and other manipulated facial content.

## Project Structure

```
├── data/                           # Dataset directory
│   ├── real_vs_fake/               # Main dataset folder
│   │   ├── real_and_fake_face/     # Contains real and fake images
│   │   │   ├── training/           # Training set
│   │   │   │   ├── real/           # Real face images for training
│   │   │   │   └── fake/           # Fake face images for training
│   │   │   └── validation/         # Validation set
│   │   │       ├── real/           # Real face images for validation
│   │   │       └── fake/           # Fake face images for validation
├── models/                         # Saved model files
├── notebooks/                      # Jupyter notebooks for exploration
├── src/                            # Source code
│   ├── data_preprocessing.py       # Data preprocessing functions
│   ├── feature_extraction.py       # Feature extraction code 
│   ├── model_training.py           # Model training scripts
│   └── evaluation.py               # Model evaluation utilities
├── saved_models/                   # Directory for saved models
│   ├── full_feature_random_forest.pkl
│   ├── selected_features_random_forest.pkl
│   ├── selected_features_gradient_boosting.pkl
│   └── selected_features_svm.pkl
├── results/                        # Results and visualizations
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```

## Features

- Image preprocessing techniques for facial analysis
- Feature extraction and selection to identify key facial authenticity markers
- Multiple machine learning classifiers (Random Forest, Gradient Boosting, SVM)
- Comprehensive model evaluation with accuracy, precision, recall, and F1-score
- Visualization of results using confusion matrices, ROC curves, and precision-recall curves

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/real-fake-face-detection.git
cd real-fake-face-detection
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection) and extract it to the `data/` directory.

## Usage

### Data Preparation

```bash
python src/data_preprocessing.py
```

This script will:
- Load the images from the dataset
- Preprocess the images (resize, normalize, etc.)
- Extract features from the facial images
- Save the processed data for model training

### Training Models

```bash
python src/model_training.py
```

This will:
- Load the preprocessed data
- Split the data into training and testing sets (80/20 split)
- Train several classifiers (Random Forest, Gradient Boosting, SVM)
- Evaluate the models and generate performance metrics
- Save the trained models to the `saved_models/` directory

### Model Evaluation

```bash
python src/evaluation.py
```

This will generate:
- Confusion matrices showing true positives, false positives, true negatives, and false negatives
- ROC curves to visualize the trade-off between true positive rate and false positive rate
- Precision-recall curves to understand model performance at different classification thresholds
- Comparison charts of different models

## Model Performance

Based on the training results:
- The trained models achieve strong performance in distinguishing between real and fake facial images
- Feature selection identified the most important visual cues for detecting manipulated images
- The models perform well on different types of fake images (printed, digital screen, etc.)

## How to Load and Use a Saved Model

```python
import joblib

# Load a trained model
model = joblib.load('saved_models/selected_features_random_forest.pkl')

# Use the model to make predictions
predictions = model.predict(X_new)
```

## Applications

This face authentication system can be applied in various domains:
- Identity verification for secure access control
- Detecting deepfakes in social media content
- Preventing identity fraud in financial services
- Enhancing security in digital communication platforms

## Future Work

- Implementation of deep learning models (CNNs, GANs) for feature extraction
- Development of a real-time face authentication system
- Expansion to handle more sophisticated deepfake techniques
- Deployment as a web or mobile application

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset provided by [CIP Lab on Kaggle](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection)
- Inspired by the growing need to detect manipulated media in the digital age
