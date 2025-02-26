import os
import json
import cv2
import numpy as np

# Map from each RGB color -> integer class ID
# class_ids: 
# 1 = Bicyclist
# 2 = Car
# 3 = Pedestrian
# 4 = MotorcycleScooter
# 5 = Truck_Bus

TARGET_CLASS_COLORS = {
    (0, 128, 192): 1,   # Bicyclist
    (64, 0, 128): 2,    # Car
    (64, 64, 0): 3,     # Pedestrian
    (192, 0, 192): 4,   # MotorcycleScooter
    (192, 128, 192): 5  # Truck_Bus
}

def get_bounding_boxes_for_class(binary_mask, class_id):
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
            "class_id": int(class_id),  # store class_id, not name
            "x_min": int(x_min),
            "x_max": int(x_max),
            "y_min": int(y_min),
            "y_max": int(y_max)
        })

    return bboxes


def generate_annotations(images_dir, masks_dir, output_json):
    """
    Generates bounding box annotations from segmentation masks, storing only class_id values.
    
    Args:
        images_dir (str): Path to the directory containing the input images.
        masks_dir (str): Path to the directory containing the corresponding mask images.
        output_json (str): Output JSON file where annotations will be saved.
        
    Returns:
        List[dict]: The list of annotation dictionaries for all processed images,
                    of the form:
                    {
                      "filename": str,
                      "width": int,
                      "height": int,
                      "bboxes": [
                         {
                           "class_id": int,
                           "x_min": int,
                           "x_max": int,
                           "y_min": int,
                           "y_max": int
                         },
                         ...
                      ]
                    }
    """
    
    all_annotations = []
    image_files = sorted(os.listdir(images_dir))
    
    for filename in image_files:
        # If not an image file, skip
        name, ext = os.path.splitext(filename)
        image_path = os.path.join(images_dir, filename)
        
        # Adjust the mask filename pattern if needed
        mask_filename = name + "_L.png"
        mask_path = os.path.join(masks_dir, mask_filename)
        
        if not os.path.exists(mask_path):
            print(f"[WARNING] Mask file not found for '{filename}'. Skipping.")
            continue
        
        # Read image and mask + Convert to RGB
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None else None
        
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) if mask is not None else None
        
        if image is None or mask is None:
            print(f"[WARNING] Could not read image or mask for '{filename}'. Skipping.")
            continue
        
        height, width = image.shape[:2]
        
        # Collect bounding boxes for all classes we care about
        image_bboxes = []
        
        for rgb_color, class_id in TARGET_CLASS_COLORS.items():
            # Create a binary mask for this class by checking if each pixel matches the color
            class_mask = cv2.inRange(mask, rgb_color, rgb_color)
            
            # Extract bounding boxes for connected components
            bboxes = get_bounding_boxes_for_class(class_mask, class_id)
            image_bboxes.extend(bboxes)
        
        annotation_data = {
            "filename": filename,
            "width": width,
            "height": height,
            "bboxes": image_bboxes
        }
        
        all_annotations.append(annotation_data)

    # Write out to JSON
    with open(output_json, "w") as f:
        json.dump(all_annotations, f, indent=2)
    
    print(f"[INFO] Annotations written to: {output_json}")
    return all_annotations

