import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw

# Load the pre-trained model
model = MobileNetV2(weights="imagenet")

def classify_image(img_array):
    """
    Classify a preprocessed image array and return predictions
    """
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    return decoded_predictions

def load_and_preprocess_image(image_path):
    """
    Load and preprocess an image for the model
    """
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

def generate_gradcam_heatmap(model, img_array, pred_index=None):
    """
    Generate Grad-CAM heatmap to identify important regions
    """
    if pred_index is None:
        pred_index = np.argmax(model.predict(img_array)[0])
    
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer('out_relu').output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, pred_index]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def get_high_attention_region(heatmap, threshold=0.7):
    """
    Get the bounding box of the high-attention region from Grad-CAM heatmap
    """
    # Resize heatmap to 224x224 to match image size
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    
    # Find regions above threshold
    high_attention = heatmap_resized > threshold
    
    # Get coordinates of high attention pixels
    coords = np.where(high_attention)
    
    if len(coords[0]) > 0:
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # Add some padding
        padding = 20
        y_min = max(0, y_min - padding)
        y_max = min(224, y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(224, x_max + padding)
        
        return (x_min, y_min, x_max, y_max)
    else:
        # If no high attention found, use center region
        return (74, 74, 150, 150)

def apply_black_box_occlusion(img_pil, region):
    """
    Occlusion Method 1: Black box over the important region
    """
    img_copy = img_pil.copy()
    draw = ImageDraw.Draw(img_copy)
    draw.rectangle(region, fill=(0, 0, 0))  # Black box
    return img_copy

def apply_blur_occlusion(img_pil, region, blur_radius=15):
    """
    Occlusion Method 2: Heavy blur over the important region
    """
    img_copy = img_pil.copy()
    
    # Extract the region to blur
    x_min, y_min, x_max, y_max = region
    region_img = img_copy.crop(region)
    
    # Apply heavy blur
    blurred_region = region_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Paste back the blurred region
    img_copy.paste(blurred_region, region)
    return img_copy

def apply_noise_occlusion(img_pil, region, noise_intensity=0.8):
    """
    Occlusion Method 3: Random noise over the important region
    """
    img_copy = img_pil.copy()
    img_array = np.array(img_copy)
    
    x_min, y_min, x_max, y_max = region
    
    # Generate random noise
    noise = np.random.randint(0, 256, (y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
    
    # Blend noise with original region
    original_region = img_array[y_min:y_max, x_min:x_max]
    blended_region = (noise_intensity * noise + (1 - noise_intensity) * original_region).astype(np.uint8)
    
    # Apply the blended region
    img_array[y_min:y_max, x_min:x_max] = blended_region
    
    return Image.fromarray(img_array)

def create_occluded_versions(img_pil, region):
    """
    Create all three types of occluded images
    """
    occlusions = {
        'Black Box': apply_black_box_occlusion(img_pil, region),
        'Heavy Blur': apply_blur_occlusion(img_pil, region),
        'Random Noise': apply_noise_occlusion(img_pil, region)
    }
    
    return occlusions

def run_occlusion_experiment(image_path):
    """
    Complete occlusion experiment pipeline
    """
    print("=== Image Occlusion Experiment ===\n")
    
    # Load and preprocess original image
    img_array, img_pil = load_and_preprocess_image(image_path)
    
    # Get original predictions
    print("Original Image Predictions:")
    original_predictions = classify_image(img_array)
    for i, (imagenet_id, label, score) in enumerate(original_predictions):
        print(f"{i + 1}: {label} ({score:.4f})")
    
    # Generate Grad-CAM to find important regions
    print("\nGenerating Grad-CAM to identify important regions...")
    heatmap = generate_gradcam_heatmap(model, img_array)
    important_region = get_high_attention_region(heatmap)
    print(f"Important region identified: {important_region}")
    
    # Create occluded versions
    print("\nCreating occluded versions...")
    occluded_images = create_occluded_versions(img_pil, important_region)
    
    # Test each occlusion
    print("\n=== Occlusion Results ===")
    results = {}
    
    for occlusion_name, occluded_img in occluded_images.items():
        print(f"\n{occlusion_name} Occlusion:")
        
        # Convert PIL to array and preprocess
        occluded_array = image.img_to_array(occluded_img.resize((224, 224)))
        occluded_array = preprocess_input(occluded_array)
        occluded_array = np.expand_dims(occluded_array, axis=0)
        
        # Get predictions
        predictions = classify_image(occluded_array)
        results[occlusion_name] = predictions
        
        for i, (imagenet_id, label, score) in enumerate(predictions):
            print(f"  {i + 1}: {label} ({score:.4f})")
        
        # Save the occluded image
        occluded_img.save(f"{occlusion_name.lower().replace(' ', '_')}_occluded.png")
    
    # Visualization
    plt.figure(figsize=(20, 12))
    
    # Original image with heatmap overlay
    plt.subplot(2, 4, 1)
    plt.imshow(img_pil)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 4, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')
    
    # Show each occluded version
    for i, (name, occluded_img) in enumerate(occluded_images.items(), 3):
        plt.subplot(2, 4, i)
        plt.imshow(occluded_img)
        plt.title(f'{name} Occlusion')
        plt.axis('off')
    
    # Performance comparison chart
    plt.subplot(2, 4, 6)
    original_top_score = original_predictions[0][2]
    occlusion_scores = [results[name][0][2] for name in occluded_images.keys()]
    
    labels = ['Original'] + list(occluded_images.keys())
    scores = [original_top_score] + occlusion_scores
    colors = ['green', 'red', 'orange', 'purple']
    
    plt.bar(labels, scores, color=colors)
    plt.title('Top Prediction Confidence Comparison')
    plt.ylabel('Confidence Score')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('occlusion_experiment_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Analysis
    print("\n=== Analysis ===")
    original_conf = original_predictions[0][2]
    print(f"Original confidence: {original_conf:.4f}")
    
    for name, predictions in results.items():
        new_conf = predictions[0][2]
        confidence_drop = original_conf - new_conf
        print(f"{name}: {new_conf:.4f} (drop: {confidence_drop:.4f})")
    
    # Find most effective occlusion
    drops = {name: original_conf - results[name][0][2] for name in results.keys()}
    most_effective = max(drops, key=drops.get)
    print(f"\nMost effective occlusion: {most_effective} (confidence drop: {drops[most_effective]:.4f})")
    
    return results

if __name__ == "__main__":
    image_path = "basic_cat.jpg"  # Replace with your image path
    run_occlusion_experiment(image_path)
