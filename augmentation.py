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

def augment_zoom(image, mask):
    """
    Zoom in to an area where rare class pixels exist. The function:
      1. Finds the bounding box of the rare pixels in the mask.
      2. Optionally, expands the bounding box by a margin.
      3. Crops the image and mask accordingly.
      4. Resizes the crop back to the full input size (720x960).
    """
    # Find coordinates where the mask has any rare color
    rare_pixels = np.zeros(mask.shape[:2], dtype=bool)
    for color in RARE_COLORS_BGR:
        match = np.all(mask == np.array(color, dtype=np.uint8), axis=-1)
        rare_pixels = rare_pixels | match

    coords = np.column_stack(np.where(rare_pixels))
    if coords.size == 0:
        # If no rare pixels found, return original resized.
        resized_img = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
        resized_mask = cv2.resize(mask, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_NEAREST)
        return resized_img, resized_mask

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Expand the bounding box by a margin (say, 20% of the bbox size)
    margin_y = int(0.2 * (y_max - y_min))
    margin_x = int(0.2 * (x_max - x_min))

    y1 = max(0, y_min - margin_y)
    y2 = min(image.shape[0], y_max + margin_y)
    x1 = max(0, x_min - margin_x)
    x2 = min(image.shape[1], x_max + margin_x)

    # Crop the region from both image and mask
    cropped_img = image[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]

    # Resize back to original input size
    zoomed_img = cv2.resize(cropped_img, (INPUT_WIDTH, INPUT_HEIGHT))
    zoomed_mask = cv2.resize(cropped_mask, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_NEAREST)
    return zoomed_img, zoomed_mask

def augment_hsv(image, mask):
    """
    Adjust the HSV values of the image.
    The mask remains unchanged.
    For example, we can shift the hue by 10 degrees, increase saturation by 10%.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Adjustments: these parameters can be tuned.
    hue_shift = -10.0        # shift hue by 10 degrees (0-180 in OpenCV)
    sat_mult = 0.8          # increase saturation by 10%
    val_mult = 1.2          # increase brightness by 10%

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
        
        # # 1. Rotation augmentation
        # rotated_img, rotated_mask = augment_rotate(image, mask)
        # save_augmented(rotated_img, rotated_mask, sample, "rotate")
        
        # 2. HSV adjustment augmentation
        hsv_img, hsv_mask = augment_hsv(image, mask)
        save_augmented(hsv_img, hsv_mask, sample, "hsv2")

if __name__ == "__main__":
    augmentation_pipeline()
