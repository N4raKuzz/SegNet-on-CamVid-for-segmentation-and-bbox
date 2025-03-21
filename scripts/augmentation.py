import os
import cv2
import numpy as np

# Directories
DATA_DIR = "data/CamVid"
TEST_IMAGES_DIR = os.path.join(DATA_DIR, "train", "images")
TEST_MASKS_DIR  = os.path.join(DATA_DIR, "train", "masks")

# Input size (height, width)
INPUT_HEIGHT = 720
INPUT_WIDTH = 960

# Rare classes in target: Note that target colors are originally defined in RGB.
# When using cv2 (which reads images in BGR), we convert:
# (0,128,192) in RGB becomes (192,128,0) in BGR -> Bicyclist
# (192,0,192) in RGB remains (192,0,192) in BGR -> MotorcycleScooter
# (192,128,192) in RGB remains (192,128,192) in BGR -> Truck_Bus
RARE_COLORS_BGR = [
    (192, 0, 192),   # MotorcycleScooter
    (192, 128, 192)    # Truck_Bus
]

def find_rare_class():
    """
    Scans through the test masks directory and selects image names 
    (without extension) for which the mask contains any rare class pixel.
    Returns a list of filenames (without extension).
    """
    selected = []
    for mask_filename in os.listdir(TEST_MASKS_DIR):
        if not mask_filename.endswith(".png"):
            continue
        mask_path = os.path.join(TEST_MASKS_DIR, mask_filename)
        mask = cv2.imread(mask_path)
        # Check if any pixel equals one of the rare colors
        found = False
        for color in RARE_COLORS_BGR:
            # Create a boolean mask where pixels equal the target color.
            match = np.all(mask == np.array(color, dtype=np.uint8), axis=-1)
            if np.any(match):
                found = True
                break
        if found:
            # Assume corresponding image filename is the same but without _L in mask name.
            # For example, if mask is 'image1_L.png', then image is 'image1.png'
            base_name = mask_filename.replace("_L.png", "").replace(".png", "")
            selected.append(base_name)
    return selected

def augment_rotate(image, mask):
    """
    Rotate both image and mask 180 degrees.
    """
    rotated_img = cv2.rotate(image, cv2.ROTATE_180)
    rotated_mask = cv2.rotate(mask, cv2.ROTATE_180)
    return rotated_img, rotated_mask

def augment_flip(image, mask, flipcode=1):
    """
    Rotate both image and mask 180 degrees.
    """

    flipped_img = cv2.flip(image, flipcode)
    flipped_mask = cv2.flip(mask, flipcode)

    return flipped_img, flipped_mask

def augment_hsv(image, mask, hue_shift, sat_mult, val_mult):
    """
    Adjust the HSV values of the image.
    The mask remains unchanged.
    For example, we can shift the hue by 10 degrees, increase saturation by 10%.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Adjustments: these parameters can be tuned.
    # hue_shift = -10.0        # shift hue by 10 degrees (0-180 in OpenCV)
    # sat_mult = 0.8          # increase saturation by 10%
    # val_mult = 1.2          # increase brightness by 10%

    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_mult, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * val_mult, 0, 255)

    adjusted_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    # For HSV adjustment, the mask remains the same.
    return adjusted_img, mask

def save_augmented(image, mask, original_name, method):
    """
    Save the augmented image and mask with the specified naming convention.
    """
    image_name = f"{original_name}_{method}.png"
    mask_name = f"{original_name}_{method}_L.png"
    image_path = os.path.join(TEST_IMAGES_DIR, image_name)
    mask_path = os.path.join(TEST_MASKS_DIR, mask_name)
    cv2.imwrite(image_path, image)
    cv2.imwrite(mask_path, mask)
    # print(f"Saved {image_path} and {mask_path}")

def augmentation_pipeline():
    """
    The pipeline to perform data augmentation on test data samples
    that contain rare classes.
    """
    rare_samples = find_rare_class()
    print(f"Found {len(rare_samples)} rare samples.")
    
    for sample in rare_samples:
        image_path = os.path.join(TEST_IMAGES_DIR, sample + ".png")
        mask_path  = os.path.join(TEST_MASKS_DIR, sample + "_L.png")
        
        # Read the image and mask
        image = cv2.imread(image_path)
        mask  = cv2.imread(mask_path)
        
        if image is None or mask is None:
            print(f"Error reading {sample}. Skipping.")
            continue
        
        # 1. Rotation augmentation
        rotated_img, rotated_mask = augment_flip(image, mask, -1)
        save_augmented(rotated_img, rotated_mask, sample, "rotate")

        # 2. Flip augmentation
        flipv_image, flipv_mask = augment_flip(image, mask, 1)
        save_augmented(flipv_image, flipv_mask, sample, 'flipv')
        fliph_image, fliph_mask = augment_flip(image, mask, 0)
        save_augmented(fliph_image, fliph_mask, sample, 'fliph')
        
        # 3. HSV adjustment augmentation
        hsv1_img, hsv1_mask = augment_hsv(image, mask, hue_shift=10, sat_mult=1.2, val_mult=1.2)
        save_augmented(hsv1_img, hsv1_mask, sample, "hsv1")
        hsv2_img, hsv2_mask = augment_hsv(image, mask, hue_shift=-10, sat_mult=1.2, val_mult=1.2)
        save_augmented(hsv2_img, hsv2_mask, sample, "hsv2")
        hsv3_img, hsv3_mask = augment_hsv(image, mask, hue_shift=10, sat_mult=0.9, val_mult=0.9)
        save_augmented(hsv3_img, hsv3_mask, sample, "hsv3")
        hsv4_img, hsv4_mask = augment_hsv(image, mask, hue_shift=-10, sat_mult=0.9, val_mult=0.9)
        save_augmented(hsv3_img, hsv3_mask, sample, "hsv4")

if __name__ == "__main__":
    augmentation_pipeline()
