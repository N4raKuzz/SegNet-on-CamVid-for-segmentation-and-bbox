import os
import cv2
import json

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

    # Color map for different class_ids.
    color_map = {
        1: (0, 0, 255),   
        2: (0, 255, 0),   
        3: (255, 0, 0),  
        4: (0, 255, 255), 
        5: (255, 255, 0)
    }
    
    # Print BBoxes
    bboxes = annotation_for_image.get("bboxes", [])
    for bbox in bboxes:
        class_id = bbox["class_id"]
        x_min = bbox["x_min"]
        x_max = bbox["x_max"]
        y_min = bbox["y_min"]
        y_max = bbox["y_max"]
        
        # Pick the color
        color = color_map.get(class_id, (255, 255, 255))
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


# Example usage:
if __name__ == "__main__":
    image_path = "./data/CamVid/test/images/0001TP_006690.png"
    mask_path  = "./data/CamVid/test/masks/0001TP_006690_L.png"
    annotation_json_path = "./data/annotations/test_annotations.json"
    
    image, mask = visualize_single_image_and_mask(image_path, mask_path, annotation_json_path)
    cv2.imshow("Visualized Image", image)
    cv2.imshow("Visualized Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
