from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
import numpy as np

def test_basic_functionality():
    """
    Test basic image processing without TensorFlow
    """
    print("=== Testing Basic Image Processing ===\n")
    
    try:
        # Test image loading
        img = Image.open("basic_cat.jpg")
        print(f"✅ Image loaded successfully: {img.size}")
        
        # Test basic filter
        img_resized = img.resize((256, 256))
        img_blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=3))
        
        # Test matplotlib
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img_resized)
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img_blurred)
        plt.title('Blurred')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('basic_test_result.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Basic image processing working!")
        print("✅ Matplotlib visualization working!")
        print("✅ File saved as 'basic_test_result.png'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_tensorflow_import():
    """
    Test if TensorFlow is available
    """
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} is available!")
        return True
    except ImportError:
        print("❌ TensorFlow not yet installed")
        return False

if __name__ == "__main__":
    print("🧪 Testing Image Processing Environment\n")
    
    # Test basic functionality
    basic_ok = test_basic_functionality()
    
    print("\n" + "="*50)
    
    # Test TensorFlow
    tf_ok = test_tensorflow_import()
    
    print("\n" + "="*50)
    print("📋 SUMMARY:")
    print(f"Basic Image Processing: {'✅ Ready' if basic_ok else '❌ Issues'}")
    print(f"TensorFlow (for AI): {'✅ Ready' if tf_ok else '⏳ Installing'}")
    
    if basic_ok and not tf_ok:
        print("\n🎯 You can start with Part 3 (Image Filters) while TensorFlow installs!")
        print("   Run: python3 enhanced_filters.py")
    elif basic_ok and tf_ok:
        print("\n🚀 Everything is ready! You can run all parts of the project!")
        print("   Part 1: python3 enhanced_classifier.py")
        print("   Part 2: python3 occlusion_experiment.py") 
        print("   Part 3: python3 enhanced_filters.py")
