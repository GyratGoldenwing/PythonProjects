from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import random

def apply_blur_filter(image_path, output_name="blurred_image.png"):
    """
    Original blur filter from basic_filter.py
    """
    try:
        img = Image.open(image_path)
        img_resized = img.resize((128, 128))
        img_blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=2))

        plt.figure(figsize=(8, 6))
        plt.imshow(img_blurred)
        plt.axis('off')
        plt.title('Gaussian Blur Filter')
        plt.savefig(output_name, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Blurred image saved as '{output_name}'.")
        return img_blurred

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def apply_edge_detection_filter(image_path, output_name="edge_detected_image.png"):
    """
    Filter 1: Edge Detection using FIND_EDGES filter
    """
    try:
        img = Image.open(image_path)
        img_resized = img.resize((256, 256))
        
        # Convert to grayscale for better edge detection
        img_gray = img_resized.convert('L')
        
        # Apply edge detection
        img_edges = img_gray.filter(ImageFilter.FIND_EDGES)
        
        # Enhance the edges
        enhancer = ImageEnhance.Contrast(img_edges)
        img_edges_enhanced = enhancer.enhance(2.0)

        plt.figure(figsize=(8, 6))
        plt.imshow(img_edges_enhanced, cmap='gray')
        plt.axis('off')
        plt.title('Edge Detection Filter')
        plt.savefig(output_name, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Edge detected image saved as '{output_name}'.")
        return img_edges_enhanced

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def apply_sharpening_filter(image_path, output_name="sharpened_image.png"):
    """
    Filter 2: Sharpening filter to enhance details
    """
    try:
        img = Image.open(image_path)
        img_resized = img.resize((256, 256))
        
        # Apply unsharp mask for sharpening
        img_sharpened = img_resized.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        
        # Additional sharpening
        enhancer = ImageEnhance.Sharpness(img_sharpened)
        img_extra_sharp = enhancer.enhance(2.0)

        plt.figure(figsize=(8, 6))
        plt.imshow(img_extra_sharp)
        plt.axis('off')
        plt.title('Sharpening Filter')
        plt.savefig(output_name, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Sharpened image saved as '{output_name}'.")
        return img_extra_sharp

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def apply_emboss_filter(image_path, output_name="embossed_image.png"):
    """
    Filter 3: Emboss filter for 3D relief effect
    """
    try:
        img = Image.open(image_path)
        img_resized = img.resize((256, 256))
        
        # Apply emboss filter
        img_embossed = img_resized.filter(ImageFilter.EMBOSS)
        
        # Enhance contrast for better effect
        enhancer = ImageEnhance.Contrast(img_embossed)
        img_embossed_enhanced = enhancer.enhance(1.5)

        plt.figure(figsize=(8, 6))
        plt.imshow(img_embossed_enhanced)
        plt.axis('off')
        plt.title('Emboss Filter')
        plt.savefig(output_name, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Embossed image saved as '{output_name}'.")
        return img_embossed_enhanced

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def apply_deep_fried_filter(image_path, output_name="deep_fried_image.png"):
    """
    Artistic Filter: 'Deep Fried' meme filter with exaggerated colors and noise
    Creates the oversaturated, high-contrast, noisy effect popular in memes
    """
    try:
        img = Image.open(image_path)
        img_resized = img.resize((256, 256))
        
        # Step 1: Increase saturation dramatically
        enhancer_color = ImageEnhance.Color(img_resized)
        img_saturated = enhancer_color.enhance(3.0)  # 300% saturation
        
        # Step 2: Increase contrast dramatically
        enhancer_contrast = ImageEnhance.Contrast(img_saturated)
        img_contrast = enhancer_contrast.enhance(2.5)  # 250% contrast
        
        # Step 3: Increase brightness slightly
        enhancer_brightness = ImageEnhance.Brightness(img_contrast)
        img_bright = enhancer_brightness.enhance(1.3)  # 130% brightness
        
        # Step 4: Add JPEG-like compression artifacts by saving and reloading
        img_bright.save("temp_compress.jpg", "JPEG", quality=10)  # Very low quality
        img_compressed = Image.open("temp_compress.jpg")
        
        # Step 5: Add random noise
        img_array = np.array(img_compressed)
        
        # Generate noise
        noise_intensity = 15
        noise = np.random.normal(0, noise_intensity, img_array.shape).astype(np.int16)
        
        # Add noise to image
        img_noisy = img_array.astype(np.int16) + noise
        img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)
        
        # Step 6: Apply slight red/orange tint (common in deep fried memes)
        img_tinted = Image.fromarray(img_noisy)
        img_array_tinted = np.array(img_tinted)
        
        # Increase red channel slightly
        img_array_tinted[:, :, 0] = np.clip(img_array_tinted[:, :, 0] * 1.2, 0, 255)
        
        # Step 7: Final sharpening for that crispy effect
        img_final = Image.fromarray(img_array_tinted.astype(np.uint8))
        img_final = img_final.filter(ImageFilter.SHARPEN)
        
        # Clean up temp file
        import os
        if os.path.exists("temp_compress.jpg"):
            os.remove("temp_compress.jpg")

        plt.figure(figsize=(8, 6))
        plt.imshow(img_final)
        plt.axis('off')
        plt.title('Deep Fried Filter 🔥')
        plt.savefig(output_name, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Deep fried image saved as '{output_name}'. 🔥🔥🔥")
        return img_final

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def compare_all_filters(image_path):
    """
    Apply all filters and create a comparison grid
    """
    print("=== Applying All Image Filters ===\n")
    
    # Apply all filters
    filters = {
        'Original': None,
        'Blur': apply_blur_filter,
        'Edge Detection': apply_edge_detection_filter,
        'Sharpening': apply_sharpening_filter,
        'Emboss': apply_emboss_filter,
        'Deep Fried 🔥': apply_deep_fried_filter
    }
    
    # Create comparison plot
    plt.figure(figsize=(18, 12))
    
    # Load original image
    original_img = Image.open(image_path).resize((256, 256))
    
    for i, (filter_name, filter_func) in enumerate(filters.items(), 1):
        plt.subplot(2, 3, i)
        
        if filter_name == 'Original':
            plt.imshow(original_img)
        else:
            # Apply filter without showing individual plots
            if filter_func:
                filtered_img = filter_func(image_path, f"{filter_name.lower().replace(' ', '_')}.png")
                if filtered_img:
                    plt.imshow(filtered_img, cmap='gray' if filter_name == 'Edge Detection' else None)
        
        plt.title(filter_name, fontsize=14, fontweight='bold')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('all_filters_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n=== Filter Analysis ===")
    print("Blur Filter: Reduces detail and noise, creates soft artistic effect")
    print("Edge Detection: Highlights boundaries and contours, good for line art")
    print("Sharpening: Enhances fine details and texture, makes image crisp")
    print("Emboss: Creates 3D relief effect, gives sculptural appearance")
    print("Deep Fried: Internet meme filter with extreme saturation and contrast 🔥")
    
    print("\nAll filters have been applied and saved as individual files!")

if __name__ == "__main__":
    image_path = "basic_cat.jpg"  # Replace with your image path
    
    print("Choose a filter to apply:")
    print("1. Blur Filter")
    print("2. Edge Detection")
    print("3. Sharpening")
    print("4. Emboss")
    print("5. Deep Fried 🔥")
    print("6. Compare All Filters")
    
    try:
        choice = input("Enter your choice (1-6) or press Enter for comparison: ").strip()
        
        if choice == "1":
            apply_blur_filter(image_path)
        elif choice == "2":
            apply_edge_detection_filter(image_path)
        elif choice == "3":
            apply_sharpening_filter(image_path)
        elif choice == "4":
            apply_emboss_filter(image_path)
        elif choice == "5":
            apply_deep_fried_filter(image_path)
        else:
            compare_all_filters(image_path)
            
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        compare_all_filters(image_path)  # Default to comparison
