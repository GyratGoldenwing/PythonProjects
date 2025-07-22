# AI Image Processing and Classification Project - Complete Guide

## Project Overview
This project explores AI image classification, visualization techniques, and creative image processing through three comprehensive parts.

## Files in This Project

### Core Files
- `base_classifier.py` - Original basic image classifier
- `basic_filter.py` - Original basic blur filter
- `basic_cat.jpg` - Sample image for testing
- `requirements.txt` - Python dependencies

### Enhanced Files (Added by Claude)
- `enhanced_classifier.py` - Classifier with Grad-CAM implementation
- `occlusion_experiment.py` - Complete image occlusion testing suite
- `enhanced_filters.py` - Multiple image filters including artistic effects

## Installation and Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Additional Dependencies for Enhanced Features:**
   ```bash
   pip install opencv-python
   ```

## Part 1: Understanding Classification and Grad-CAM

### What Each Line Does in base_classifier.py:

**Imports and Setup:**
- `import tensorflow as tf` - Machine learning framework
- `tf.get_logger().setLevel('ERROR')` - Suppress warnings
- `from tensorflow.keras.applications import MobileNetV2` - Pre-trained model
- `from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions` - Model-specific functions
- `import numpy as np` - Numerical operations
- `import matplotlib.pyplot as plt` - Plotting and visualization

**Model Loading:**
- `model = MobileNetV2(weights="imagenet")` - Loads pre-trained model with ImageNet weights

**Image Processing:**
- `img = image.load_img(image_path, target_size=(224, 224))` - Loads and resizes image
- `img_array = image.img_to_array(img)` - Converts to numerical array
- `img_array = preprocess_input(img_array)` - Normalizes values for model
- `img_array = np.expand_dims(img_array, axis=0)` - Adds batch dimension

**Classification:**
- `predictions = model.predict(img_array)` - Gets model predictions
- `decoded_predictions = decode_predictions(predictions, top=3)[0]` - Converts to readable labels

### Running the Enhanced Classifier:
```bash
python enhanced_classifier.py
```

**What is Grad-CAM?**
Grad-CAM (Gradient-weighted Class Activation Mapping) shows which parts of an image the AI focuses on:
- Uses gradients from the final prediction back to the last convolutional layer
- Creates a heatmap where red areas = high importance, blue areas = low importance
- Helps understand AI decision-making process

## Part 2: Image Occlusion Experiments

### Three Occlusion Methods:

1. **Black Box Occlusion:** Places solid black rectangles over important regions
2. **Heavy Blur Occlusion:** Applies strong Gaussian blur to hide details
3. **Random Noise Occlusion:** Adds random pixel noise to disrupt recognition

### Running the Occlusion Experiment:
```bash
python occlusion_experiment.py
```

**Expected Results:**
- Original image should have high confidence predictions
- Occluded images should show reduced confidence
- Different occlusion methods will impact performance differently
- The method that targets the most important visual features will cause the biggest drop

## Part 3: Creative Image Filtering

### What Each Line Does in basic_filter.py:

**Imports:**
- `from PIL import Image, ImageFilter` - Python Imaging Library for image processing
- `import matplotlib.pyplot as plt` - For displaying results

**Filter Application:**
- `img = Image.open(image_path)` - Opens the image file
- `img_resized = img.resize((128, 128))` - Resizes to smaller dimensions
- `img_blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=2))` - Applies blur effect
- `plt.imshow(img_blurred)` - Displays the result
- `plt.savefig("blurred_image.png")` - Saves to file

### Enhanced Filters Available:

1. **Blur Filter:** Soft, dreamy effect using Gaussian blur
2. **Edge Detection:** Highlights boundaries and contours
3. **Sharpening:** Enhances fine details and textures
4. **Emboss:** Creates 3D relief sculptural effect
5. **Deep Fried 🔥:** Internet meme filter with extreme saturation, contrast, and noise

### Running the Enhanced Filters:
```bash
python enhanced_filters.py
```

**Deep Fried Filter Process:**
1. Dramatically increase color saturation (300%)
2. Boost contrast significantly (250%)
3. Add compression artifacts (low-quality JPEG)
4. Inject random noise for grittiness
5. Apply red/orange tint
6. Final sharpening for "crispy" effect

## Running the Complete Project

### Step-by-Step Execution:

1. **Test Basic Classifier:**
   ```bash
   python base_classifier.py
   ```
   Record the top-3 predictions and confidence scores.

2. **Run Enhanced Classifier with Grad-CAM:**
   ```bash
   python enhanced_classifier.py
   ```
   Analyze which parts of the image the model focuses on.

3. **Conduct Occlusion Experiments:**
   ```bash
   python occlusion_experiment.py
   ```
   Compare how different occlusion methods affect classification performance.

4. **Explore Image Filters:**
   ```bash
   python enhanced_filters.py
   ```
   Try different artistic effects and create your own custom filter.

## Analysis Questions to Consider

### Part 1 - Classification & Grad-CAM:
- Do the AI's explanations of the code make sense?
- What objects or features does the Grad-CAM highlight?
- Are these the same features you would focus on?

### Part 2 - Occlusion:
- Which occlusion method most severely impacts classification?
- Does occluding the Grad-CAM highlighted areas reduce confidence?
- What does this tell us about how the AI "sees" images?

### Part 3 - Filters:
- How do different filters change the artistic mood of the image?
- What creative effects can you achieve with the Deep Fried filter?
- Can you combine multiple filters for unique effects?

## Expected Output Files

After running all experiments, you should have:
- `gradcam_results.png` - Grad-CAM visualization
- `black_box_occluded.png` - Black box occlusion result
- `heavy_blur_occluded.png` - Blur occlusion result
- `random_noise_occluded.png` - Noise occlusion result
- `occlusion_experiment_results.png` - Complete occlusion analysis
- `all_filters_comparison.png` - Filter comparison grid
- Individual filter outputs for each effect

## Troubleshooting

### Common Issues:
1. **ModuleNotFoundError:** Run `pip install -r requirements.txt`
2. **OpenCV Error:** Install with `pip install opencv-python`
3. **Memory Issues:** Reduce image size or use smaller batch sizes
4. **File Not Found:** Ensure `basic_cat.jpg` is in the project directory

### Performance Tips:
- Use smaller images for faster processing
- Close matplotlib windows to free memory
- Run experiments one at a time for large images

## Project Reflection

This project demonstrates:
- How AI models make decisions through Grad-CAM visualization
- The importance of specific visual features for classification
- Creative applications of computer vision techniques
- The intersection of AI understanding and human artistic expression

The Deep Fried filter specifically shows how we can use AI tools to create internet culture artifacts, bridging technical image processing with contemporary digital art forms.
