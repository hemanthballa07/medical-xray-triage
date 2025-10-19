"""
Generate realistic synthetic chest X-ray images for testing.

This script creates more diverse and realistic-looking chest X-ray images
with various anatomical features and potential abnormalities.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random

def create_chest_outline(width, height):
    """Create a realistic chest outline."""
    # Create base image
    img = Image.new('RGB', (width, height), color='black')
    draw = ImageDraw.Draw(img)
    
    # Chest cavity outline
    chest_width = int(width * 0.7)
    chest_height = int(height * 0.8)
    chest_x = (width - chest_width) // 2
    chest_y = int(height * 0.15)
    
    # Draw chest outline with slight curvature
    chest_points = [
        (chest_x, chest_y),
        (chest_x + chest_width, chest_y),
        (chest_x + chest_width, chest_y + chest_height),
        (chest_x, chest_y + chest_height)
    ]
    
    # Draw ribs
    rib_color = (80, 80, 80)
    for i in range(8):
        y = chest_y + int(chest_height * 0.2) + i * int(chest_height * 0.08)
        # Curved ribs
        for x in range(chest_x + 20, chest_x + chest_width - 20, 5):
            # Add some randomness for natural look
            offset = random.randint(-3, 3)
            draw.point((x, y + offset), fill=rib_color)
    
    return img

def add_lungs(img, is_abnormal=False):
    """Add lung fields to the chest image."""
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    # Lung areas
    left_lung_x = int(width * 0.25)
    right_lung_x = int(width * 0.55)
    lung_y = int(height * 0.25)
    lung_width = int(width * 0.2)
    lung_height = int(height * 0.6)
    
    # Normal lung tissue (darker areas)
    lung_color = (60, 60, 60)
    
    # Left lung
    draw.ellipse([left_lung_x, lung_y, left_lung_x + lung_width, lung_y + lung_height], 
                 fill=lung_color, outline=(40, 40, 40))
    
    # Right lung
    draw.ellipse([right_lung_x, lung_y, right_lung_x + lung_width, lung_y + lung_height], 
                 fill=lung_color, outline=(40, 40, 40))
    
    if is_abnormal:
        # Add abnormalities
        add_abnormalities(img, left_lung_x, right_lung_x, lung_y, lung_width, lung_height)
    
    return img

def add_abnormalities(img, left_lung_x, right_lung_x, lung_y, lung_width, lung_height):
    """Add various types of abnormalities to the lungs."""
    draw = ImageDraw.Draw(img)
    
    # Randomly choose abnormality type
    abnormality_type = random.choice(['opacity', 'consolidation', 'nodule', 'effusion'])
    
    if abnormality_type == 'opacity':
        # Ground glass opacity
        opacity_color = (120, 120, 120)
        opacity_x = random.choice([left_lung_x, right_lung_x]) + random.randint(10, 30)
        opacity_y = lung_y + random.randint(50, 100)
        opacity_size = random.randint(40, 80)
        
        # Create semi-transparent opacity
        opacity_img = Image.new('RGBA', (opacity_size, opacity_size), (0, 0, 0, 0))
        opacity_draw = ImageDraw.Draw(opacity_img)
        opacity_draw.ellipse([0, 0, opacity_size, opacity_size], fill=(120, 120, 120, 100))
        
        # Paste with transparency
        img.paste(opacity_img, (opacity_x, opacity_y), opacity_img)
        
    elif abnormality_type == 'consolidation':
        # Airspace consolidation
        consolidation_color = (140, 140, 140)
        consolidation_x = random.choice([left_lung_x, right_lung_x]) + random.randint(5, 25)
        consolidation_y = lung_y + random.randint(80, 120)
        consolidation_width = random.randint(60, 100)
        consolidation_height = random.randint(30, 60)
        
        draw.ellipse([consolidation_x, consolidation_y, 
                     consolidation_x + consolidation_width, 
                     consolidation_y + consolidation_height], 
                     fill=consolidation_color)
        
    elif abnormality_type == 'nodule':
        # Pulmonary nodule
        nodule_color = (100, 100, 100)
        nodule_x = random.choice([left_lung_x, right_lung_x]) + random.randint(20, 40)
        nodule_y = lung_y + random.randint(60, 150)
        nodule_size = random.randint(15, 30)
        
        draw.ellipse([nodule_x, nodule_y, nodule_x + nodule_size, nodule_y + nodule_size], 
                     fill=nodule_color)
        
    elif abnormality_type == 'effusion':
        # Pleural effusion
        effusion_color = (160, 160, 160)
        effusion_x = random.choice([left_lung_x, right_lung_x]) + random.randint(10, 30)
        effusion_y = lung_y + lung_height - random.randint(40, 80)
        effusion_width = random.randint(40, 70)
        effusion_height = random.randint(20, 40)
        
        draw.ellipse([effusion_x, effusion_y, 
                     effusion_x + effusion_width, 
                     effusion_y + effusion_height], 
                     fill=effusion_color)

def add_heart_shadow(img):
    """Add heart shadow to the image."""
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    # Heart shadow (left side of chest)
    heart_x = int(width * 0.2)
    heart_y = int(height * 0.3)
    heart_width = int(width * 0.15)
    heart_height = int(height * 0.4)
    
    heart_color = (70, 70, 70)
    draw.ellipse([heart_x, heart_y, heart_x + heart_width, heart_y + heart_height], 
                 fill=heart_color, outline=(50, 50, 50))
    
    return img

def add_anatomical_features(img):
    """Add other anatomical features like spine, diaphragm."""
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    # Spine
    spine_x = width // 2
    spine_color = (90, 90, 90)
    spine_width = 8
    
    # Draw spine as vertical line with some curvature
    for y in range(int(height * 0.2), int(height * 0.9), 3):
        x_offset = random.randint(-2, 2)
        draw.rectangle([spine_x - spine_width//2 + x_offset, y, 
                       spine_x + spine_width//2 + x_offset, y + 2], 
                       fill=spine_color)
    
    # Diaphragm
    diaphragm_color = (80, 80, 80)
    diaphragm_y = int(height * 0.75)
    
    # Draw curved diaphragm
    for x in range(int(width * 0.1), int(width * 0.9), 5):
        y_offset = random.randint(-5, 5)
        draw.point((x, diaphragm_y + y_offset), fill=diaphragm_color)
    
    return img

def add_noise_and_texture(img):
    """Add realistic noise and texture to the image."""
    # Convert to numpy for noise addition
    img_array = np.array(img)
    
    # Add gaussian noise
    noise = np.random.normal(0, 10, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    # Add some texture variations
    texture_noise = np.random.normal(0, 5, img_array.shape)
    img_array = np.clip(img_array + texture_noise, 0, 255).astype(np.uint8)
    
    # Convert back to PIL
    img = Image.fromarray(img_array)
    
    # Apply slight blur for realistic effect
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    return img

def generate_realistic_xray(is_abnormal=False, width=320, height=320):
    """Generate a realistic chest X-ray image."""
    # Set random seed for reproducibility
    random.seed(42 if not is_abnormal else 123)
    np.random.seed(42 if not is_abnormal else 123)
    
    # Create base image
    img = create_chest_outline(width, height)
    
    # Add anatomical features
    img = add_lungs(img, is_abnormal=is_abnormal)
    img = add_heart_shadow(img)
    img = add_anatomical_features(img)
    
    # Add noise and texture
    img = add_noise_and_texture(img)
    
    return img

def main():
    """Generate a set of realistic chest X-ray images."""
    # Create output directory
    output_dir = "data/sample/images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate normal images
    print("Generating normal chest X-ray images...")
    for i in range(5):
        img = generate_realistic_xray(is_abnormal=False, width=320, height=320)
        img.save(os.path.join(output_dir, f"normal_realistic_{i+1:03d}.png"))
        print(f"  Generated: normal_realistic_{i+1:03d}.png")
    
    # Generate abnormal images
    print("\nGenerating abnormal chest X-ray images...")
    for i in range(5):
        img = generate_realistic_xray(is_abnormal=True, width=320, height=320)
        img.save(os.path.join(output_dir, f"abnormal_realistic_{i+1:03d}.png"))
        print(f"  Generated: abnormal_realistic_{i+1:03d}.png")
    
    # Update labels.csv
    print("\nUpdating labels.csv...")
    labels_data = []
    
    # Add original simple images
    labels_data.extend([
        ("images/normal_001.png", 0),
        ("images/normal_002.png", 0),
        ("images/abnormal_001.png", 1),
        ("images/abnormal_002.png", 1)
    ])
    
    # Add new realistic images
    for i in range(5):
        labels_data.append((f"images/normal_realistic_{i+1:03d}.png", 0))
        labels_data.append((f"images/abnormal_realistic_{i+1:03d}.png", 1))
    
    # Write labels.csv
    import pandas as pd
    df = pd.DataFrame(labels_data, columns=['filepath', 'label'])
    df.to_csv('data/sample/labels.csv', index=False)
    
    print(f"\nGenerated {len(labels_data)} total images:")
    print(f"  Normal images: {len([x for x in labels_data if x[1] == 0])}")
    print(f"  Abnormal images: {len([x for x in labels_data if x[1] == 1])}")
    print(f"\nLabels saved to: data/sample/labels.csv")

if __name__ == "__main__":
    main()

