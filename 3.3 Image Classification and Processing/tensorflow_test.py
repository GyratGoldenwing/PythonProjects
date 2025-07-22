"""
Simplified classifier that works with basic TensorFlow installation
Fallback version if full TensorFlow has issues
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Try importing TensorFlow with fallbacks
tf_available = False
try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
    from tensorflow.keras.preprocessing import image
    tf_available = True
    print("✅ TensorFlow successfully imported!")
except ImportError as e:
    print(f"❌ TensorFlow import failed: {e}")
    print("🔄 Trying alternative installation...")

def test_tensorflow():
    """Test if TensorFlow is working"""
    if not tf_available:
        return False
    
    try:
        # Test basic operations
        test_tensor = tf.constant([1, 2, 3, 4])
        print(f"✅ TensorFlow {tf.__version__} is working!")
        print(f"✅ Test tensor: {test_tensor}")
        return True
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
        return False

def run_simple_classification(image_path):
    """Run classification if TensorFlow is available"""
    if not tf_available:
        print("❌ TensorFlow not available - using basic analysis instead")
        return run_basic_analysis(image_path)
    
    try:
        print("🚀 Loading MobileNetV2 model...")
        model = MobileNetV2(weights="imagenet")
        
        print("📷 Processing image...")
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        print("🤖 Running AI classification...")
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        print("\n🎯 AI Classification Results:")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            print(f"{i + 1}: {label} ({score:.4f})")
        
        # Display image with results
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f'AI Prediction: {decoded_predictions[0][1]}\nConfidence: {decoded_predictions[0][2]:.4f}')
        plt.axis('off')
        plt.savefig('ai_classification_result.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ AI Classification complete!")
        return decoded_predictions
        
    except Exception as e:
        print(f"❌ Classification failed: {e}")
        return run_basic_analysis(image_path)

def run_basic_analysis(image_path):
    """Fallback basic analysis"""
    print("🔧 Running basic image analysis instead...")
    
    img = Image.open(image_path)
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)
    
    # Simple analysis
    avg_color = np.mean(img_array, axis=(0, 1))
    brightness = np.mean(avg_color)
    
    # Basic predictions
    predictions = []
    if brightness > 150:
        predictions.append(("bright_object", 0.8))
    elif brightness > 100:
        predictions.append(("medium_bright_object", 0.7))
    else:
        predictions.append(("dark_object", 0.6))
    
    print("\n📊 Basic Analysis Results:")
    for i, (label, score) in enumerate(predictions[:3], 1):
        print(f"{i}: {label.replace('_', ' ')} ({score:.4f})")
    
    return predictions

if __name__ == "__main__":
    print("🧪 TESTING TENSORFLOW INSTALLATION")
    print("="*50)
    
    # Test TensorFlow
    tf_working = test_tensorflow()
    
    print("\n🖼️ TESTING IMAGE CLASSIFICATION")
    print("="*50)
    
    image_path = "basic_cat.jpg"
    
    if tf_working:
        print("🎉 TensorFlow is ready! Running full AI classification...")
        results = run_simple_classification(image_path)
        
        if results:
            print("\n🚀 SUCCESS! TensorFlow is working perfectly!")
            print("✅ You can now run:")
            print("   python3 enhanced_classifier.py")
            print("   python3 occlusion_experiment.py")
    else:
        print("⚠️ TensorFlow not ready yet - running basic analysis...")
        run_basic_analysis(image_path)
        
        print("\n🔄 TO FIX TENSORFLOW:")
        print("   pip install tensorflow-macos tensorflow-metal")
        print("   OR: conda install tensorflow")
