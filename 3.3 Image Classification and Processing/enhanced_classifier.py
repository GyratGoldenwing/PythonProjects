import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the pre-trained model
model = MobileNetV2(weights="imagenet")

def classify_image(image_path):
    """
    Classify an image using MobileNetV2 and return predictions
    """
    try:
        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Make predictions
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        print("Top-3 Predictions:")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            print(f"{i + 1}: {label} ({score:.4f})")
        
        return img_array, predictions, decoded_predictions

    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None, None

def generate_gradcam_heatmap(model, img_array, pred_index=None):
    """
    Generate Grad-CAM heatmap for the given image
    """
    # Get the index of the top prediction if not specified
    if pred_index is None:
        pred_index = np.argmax(model.predict(img_array)[0])
    
    # Create a model that maps the input image to the activations of the last conv layer
    # as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer('out_relu').output, model.output]  # MobileNetV2's last conv layer
    )
    
    # Compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, pred_index]
    
    # Gradient of the output neuron (top predicted class) with respect to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply each channel in the feature map array by "how important this channel is"
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def display_gradcam(image_path, heatmap, alpha=0.4):
    """
    Display the original image with Grad-CAM heatmap overlay
    """
    # Load the original image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    
    # Use jet colormap to colorize heatmap
    jet = plt.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((224, 224))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
    
    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
    
    # Display the results
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')
    
    # Superimposed
    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    plt.title('Grad-CAM Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('gradcam_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return superimposed_img

def analyze_image_with_gradcam(image_path):
    """
    Complete analysis: classify image and generate Grad-CAM visualization
    """
    print("=== Image Classification with Grad-CAM Analysis ===\n")
    
    # Classify the image
    img_array, predictions, decoded_predictions = classify_image(image_path)
    
    if img_array is not None:
        # Generate Grad-CAM heatmap for the top prediction
        heatmap = generate_gradcam_heatmap(model, img_array)
        
        # Display results
        print(f"\nGenerating Grad-CAM visualization for top prediction: {decoded_predictions[0][1]}")
        superimposed_img = display_gradcam(image_path, heatmap)
        
        print("\nGrad-CAM Analysis Complete!")
        print("The heatmap shows which parts of the image the model focused on for classification.")
        
        return decoded_predictions, heatmap
    
    return None, None

if __name__ == "__main__":
    image_path = "basic_cat.jpg"  # Replace with your image path
    analyze_image_with_gradcam(image_path)
