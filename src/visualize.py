import os
import cv2
import json
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

TARGET_CLASS_COLORS = {
    (0, 128, 192): 1,   # Bicyclist
    (64, 0, 128): 2,    # Car
    (64, 64, 0): 3,     # Pedestrian
    (192, 0, 192): 4,   # MotorcycleScooter
    (192, 128, 192): 5  # Truck_Bus
}

TARGET_CLASS_ID = {
    1: "Bicyclist",
    2: "Car",
    3: "Pedestrian",
    4: "MotorcycleScooter",
    5: "Truck_Bus"
}

CLASS_COLOR_MAPPING = {0: (0, 0, 0)}
for color, label in TARGET_CLASS_COLORS.items():
    CLASS_COLOR_MAPPING[label] = color

def visualize_single_image_and_mask(image_path, mask_path, annotation_json_path):
    """
    1. Reads the entire annotation JSON.
    2. Finds the annotation entry for the single given image filename.
    3. Loads the image and mask.
    4. Draws bounding boxes (color-coded by class_id) on both image and mask.
    5. Prints the bbox coordinates.

    Args:
        image_path (str): Path to the image.
        mask_path (str): Path to the mask.
        annotation_json_path (str): Path to the entire annotation JSON file.
    """
    # 1) Load the entire annotations JSON
    with open(annotation_json_path, "r") as f:
        annotations = json.load(f)
    image_filename = os.path.basename(image_path) 
    # Find the annotation entry for this filename
    annotation_for_image = None
    for ann in annotations:
        if ann["filename"] == image_filename:
            annotation_for_image = ann
            break
    
    if annotation_for_image is None:
        print(f"[WARNING] No annotation found in '{annotation_json_path}' for '{image_filename}'.")
        return
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None else None
    
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) if mask is not None else None

    # Print BBoxes
    bboxes = annotation_for_image.get("bboxes", [])
    for bbox in bboxes:
        class_id = bbox["class_id"]
        x_min = bbox["x_min"]
        x_max = bbox["x_max"]
        y_min = bbox["y_min"]
        y_max = bbox["y_max"]
        
        # Pick the color
        color = TARGET_CLASS_COLORS.get(class_id, (255, 255, 255))
        thickness = 2
        
        # Draw on the image
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        cv2.putText(image, f"ID:{class_id}", (x_min, max(y_min - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw on the mask
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), color, thickness)
        cv2.putText(mask, f"ID:{class_id}", (x_min, max(y_min - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Print bbox coordinates to the console
        print(f"Class ID: {class_id}, BBox => (x_min={x_min}, x_max={x_max}, "
              f"y_min={y_min}, y_max={y_max})")
        
    return image, mask

def colorize_mask(mask):
    """
    Convert a segmentation mask (with class labels) into a color image.
    This is used for the predicted mask.
    
    :param mask: numpy array of shape (H, W) with integer labels.
    :return: numpy array of shape (H, W, 3) with RGB colors.
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in CLASS_COLOR_MAPPING.items():
        color_mask[mask == cls] = color
    return color_mask

def colorize_target_mask(mask):
    """
    Convert a target segmentation mask (with class ids) into a color image using TARGET_CLASS_COLORS.
    
    :param mask: numpy array of shape (H, W) with integer labels.
    :return: numpy array of shape (H, W, 3) with RGB colors.
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    # Invert the mapping: from class id to color.
    target_id_to_color = {class_id: color for color, class_id in TARGET_CLASS_COLORS.items()}
    for class_id, color in target_id_to_color.items():
        color_mask[mask == class_id] = color
    return color_mask

def visualize_predictions(model, dataset, device, num_samples=10):
    """
    Randomly select samples from the dataset, run inference, and save a figure 
    with two subplots per sample:
      - The first subplot shows the original image with the target mask overlay.
      - The second subplot shows the original image with the predicted mask overlay.
    A figure-level legend is added at the top, displaying the color-class-classid mapping.
    The resulting figure is saved in the "results" folder.
    """
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)
    plt.figure(figsize=(10, 5 * num_samples))
    
    # Create an inverted mapping for legend: class id -> color.
    target_id_to_color = {class_id: color for color, class_id in TARGET_CLASS_COLORS.items()}
    
    for i, idx in enumerate(indices):
        image, target_mask = dataset[idx]  # includes ground-truth mask
        # The image is a tensor of shape (C, H, W). Add batch dim.
        input_tensor = image.unsqueeze(0).to(device)

        # Get predicted mask.
        with torch.no_grad():
            seg_logits = model(input_tensor)
            pred_mask = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy()

        # Colorize predicted mask.
        pred_mask_color = colorize_mask(pred_mask)
        # Colorize target mask.
        colored_target_mask = colorize_target_mask(target_mask)
        # Convert image tensor to numpy array (H, W, C) for visualization.
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np

        # First subplot: original image with target mask overlay.
        ax1 = plt.subplot(num_samples, 2, 2 * i + 1)
        ax1.imshow(img_np)
        ax1.imshow(colored_target_mask, alpha=0.5)
        ax1.set_title("Target Mask Overlay")
        ax1.axis("off")

        # Second subplot: original image with predicted mask overlay.
        ax2 = plt.subplot(num_samples, 2, 2 * i + 2)
        ax2.imshow(img_np)
        ax2.imshow(pred_mask_color, alpha=0.5)
        ax2.set_title("Predicted Mask Overlay")
        ax2.axis("off")
    
    # Create legend patches with line breaks in the label.
    patches = []
    for class_id, class_name in TARGET_CLASS_ID.items():
        color = target_id_to_color.get(class_id, (0, 0, 0))
        normalized_color = tuple([c / 255 for c in color])
        # Add \n for multi-line labels
        label = f"{color}\n{class_name} - {class_id}"
        patches.append(mpatches.Patch(color=normalized_color, label=label))
    
    # Add a figure-level legend at the top (with extra top margin).
    # 'bbox_to_anchor' moves the legend, 'ncol' sets how many patches per row.
    plt.figlegend(
        handles=patches, 
        loc='upper center', 
        ncol=len(patches), 
        bbox_to_anchor=(0.5, 1.15)
    )
    
    # Increase the top margin so the legend is not clipped.
    plt.subplots_adjust(top=0.80)
    
    # Ensure the 'results' directory exists.
    if not os.path.exists("results"):
        os.makedirs("results")
    
    # Save the figure.
    plt.savefig("results/predictions.png")
    plt.close()