# Sand Grain Analysis Pipeline

An end-to-end pipeline for beach sand grain size distribution estimation and classification among three categories: Dune, Intertidal, and Berm.

## Features

### Implemented Features

#### Preprocessing Module (preprocess.py)

- ArUco marker-based image reorientation for consistent image alignment
- Pixel-to-length mapping using standard reference object (ArUco marker)
- Illumination correction using Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Haze removal using Dark Channel Prior method
- Image standardization to uniform dimensions

#### Classification Module (model.py, train.py)

- Decision tree-based classification for sand type (Dune/Intertidal/Berm)
- Feature extraction:
  - Color features from HSV color space
  - Basic grain size distribution parameters
- Grid search cross-validation for hyperparameter optimization
- Periodic model retraining capability
- Model persistence and versioning

### _Planned Features (To Be Implemented)_

#### Feature Extraction Module

- Background-Foreground analysis and masking
- Advanced grain instance detection using watershed algorithm
- Convolution-based image segmentation enhancement
- Detailed grain distribution density estimation
- Shape analysis of individual grains
- Texture feature extraction
- Statistical distribution analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/Nikhil-iitg27/SandGrainAnalysis.git
cd SandGrainAnalysis

# Install required packages
pip install numpy opencv-python scikit-learn tensorflow joblib
```

## Project Structure

```
SandGrainAnalysis/
├── preProcess/
│   └── preprocess.py      # Image preprocessing module
├── featureExtract/        # (Planned)
│   ├── model.py
│   ├── train.py
│   └── test.py
├── classificationModel/
│   ├── model.py          # Classification model definition
│   ├── train.py         # Training pipeline
│   └── weights/         # Directory for model weights
└── run.py               # Main execution script
```

## Usage

### Training Mode

Train the classification model with your dataset:

```bash
python run.py --mode train \
              --input /path/to/training/data \
              --model-dir classificationModel/weights \
              --accuracy-threshold 0.85
```

Training data should be organized in the following structure:

```
training_data/
├── dune/
│   ├── image1.jpg
│   └── image2.jpg
├── intertidal/
│   ├── image3.jpg
│   └── image4.jpg
└── berm/
    ├── image5.jpg
    └── image6.jpg
```

### Prediction Mode

Analyze a single sand image:

```bash
python run.py --mode predict \
              --input /path/to/image.jpg \
              --model-dir classificationModel/weights
```

## Requirements

- Python 3.8+
- OpenCV (cv2)
- NumPy
- scikit-learn
- TensorFlow
- joblib

## Image Requirements

- Images should contain at least one ArUco marker (4x4, dictionary 50) for scale reference
- Good lighting conditions recommended for optimal results
- Minimal background noise or interference
- Recommended resolution: 1920x1080 or higher

## Further Enhancements Aiming For

1. **Advanced Grain Analysis**

   - Individual grain boundary detection
   - Shape factor calculation
   - Grain orientation analysis
   - 3D surface reconstruction

2. **Machine Learning Improvements**

   - Deep learning-based grain detection
   - Ensemble methods for classification
   - Uncertainty estimation
   - Active learning for model improvement

3. **User Interface**

   - Web-based interface
   - Batch processing capabilities
   - Real-time analysis
   - Result visualization and reporting

4. **Data Management**
   - Database integration
   - Historical analysis
   - Data export formats
   - Automated reporting

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Citations

For the implemented algorithms and methods:

1. ArUco Marker Detection:

   - Garrido-Jurado, S., et al. (2014). Automatic generation and detection of highly reliable fiducial markers under occlusion.

2. Dark Channel Prior for Haze Removal:

   - He, K., Sun, J., & Tang, X. (2011). Single image haze removal using dark channel prior.

3. Watershed Algorithm (Planned):
   - Beucher, S., & Meyer, F. (1993). The morphological approach to segmentation: the watershed transformation.
