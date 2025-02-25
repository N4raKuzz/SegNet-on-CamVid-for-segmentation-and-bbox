import os
import json
import cv2
import numpy as np


TARGET_CLASS_COLORS = {
    (0, 128, 192): "Bicyclist",         
    (64, 0, 128): "Car",              
    (64, 64, 0): "Pedestrian", 
    (192, 0, 192): "MotorcycleScooter",
    (192, 128, 192): "Truck_Bus" 
}

def get_bounding_boxes_for_class(binary_mask, class_name):
        """
        Given a binary mask (mask == 255 => that pixel belongs to the class),
        find connected components and return bounding boxes for each component.
        """
        # Connected-component labeling with stats
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        
        bboxes = []
        # Label 0 is background; skip it
        for label_idx in range(1, num_labels):
            left   = stats[label_idx, cv2.CC_STAT_LEFT]
            top    = stats[label_idx, cv2.CC_STAT_TOP]
            width  = stats[label_idx, cv2.CC_STAT_WIDTH]
            height = stats[label_idx, cv2.CC_STAT_HEIGHT]
            
            x_min = left
            x_max = left + width - 1
            y_min = top
            y_max = top + height - 1
            
            bboxes.append({
                "class": class_name,
                "x_min": int(x_min),
                "x_max": int(x_max),
                "y_min": int(y_min),
                "y_max": int(y_max)
            })
        
        return bboxes
    

def generate_annotations(images_dir, masks_dir, output_json):
    """
    Generates bounding box annotations from segmentation masks.
    
    Args:
        images_dir (str): Path to the directory containing the input images.
        masks_dir (str): Path to the directory containing the corresponding mask images.
        output_json (str): Output JSON file where annotations will be saved.
        
    Returns:
        List of dict: The list of annotation dictionaries for all processed images.
                     Each entry is of the form:
                     {
                       "filename": str,
                       "width": int,
                       "height": int,
                       "bboxes": [
                          {
                            "class": str,
                            "x_min": int,
                            "x_max": int,
                            "y_min": int,
                            "y_max": int
                          },
                          ...
                       ]
                     }
    """
    
    # The set of classes we actually care about
    target_classes = set(TARGET_CLASS_COLORS.keys())

    # Main logic: scan images, match masks, extract bounding boxes.
    all_annotations = []
    
    valid_exts = {".png", ".jpg", ".jpeg"}
    image_files = sorted(os.listdir(images_dir))
    
    for filename in image_files:
        name, ext = os.path.splitext(filename)
        if ext.lower() not in valid_exts:
            continue
        
        # Full path to the image
        image_path = os.path.join(images_dir, filename)
        
        # Assume mask has the same name but in the masks_dir
        mask_filename = name + ".png"
        mask_path = os.path.join(masks_dir, mask_filename)
        
        if not os.path.exists(mask_path):
            print(f"[WARNING] Mask file not found for {filename}. Skipping.")
            continue
        
        # Read image and mask
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)
        
        if image is None or mask is None:
            print(f"[WARNING] Could not read image or mask for {filename}. Skipping.")
            continue
        
        height, width = image.shape[:2]
        
        # Collect bounding boxes for all classes we care about
        image_bboxes = []
        
        for class_name, bgr_color in TARGET_CLASS_COLORS.items():
            # Create a binary mask for this class by checking if each pixel matches the color
            class_mask = cv2.inRange(mask, bgr_color, bgr_color)  # 255 where exactly matches bgr_color
            
            # Extract bounding boxes for connected components
            bboxes = get_bounding_boxes_for_class(class_mask, class_name)
            image_bboxes.extend(bboxes)
        
        annotation_data = {
            "filename": filename,
            "width": width,
            "height": height,
            "bboxes": image_bboxes
        }
        
        all_annotations.append(annotation_data)

    with open(output_json, "w") as f:
        json.dump(all_annotations, f, indent=2)
    
    print(f"[INFO] Annotations written to: {output_json}")
    return all_annotations


