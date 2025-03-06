import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn

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

TRAIN_TO_EVAL_MAPPING = {
    2: 1,   # Bicyclist
    5: 2,   # Car
    16: 3,  # Pedestrian
    13: 4,  # MotorcycleScooter
    27: 5   # Truck_Bus
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


def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two boxes.
    Boxes are in format [x_min, y_min, x_max, y_max].
    """
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])
    inter_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area + 1e-6
    return inter_area / union_area

def evaluate_segmentation(model, dataloader, device):
    """
    Evaluate segmentation performance by computing mean IoU for the desired target classes.
    
    The model outputs predictions over 32 classes. Using `target_mapping`, we map these predictions
    to the 5 evaluation classes. Ground truth masks should already be encoded in the evaluation 
    domain (e.g., using your TEST_CLASS_COLORS mapping).

    :param model: the segmentation model.
    :param dataloader: DataLoader providing (image, mask) pairs.
    :param device: computation device 
    :return: mean IoU (float) and IoU per evaluation class (numpy array).
    """
    model.eval()
    
    # Determine the evaluation class ids (e.g., [1, 2, 3, 4, 5])
    eval_class_ids = sorted(set(TRAIN_TO_EVAL_MAPPING.values()))
    
    # Initialize intersection and union accumulators for each evaluation class.
    total_intersections = {cls: 0 for cls in eval_class_ids}
    total_unions = {cls: 0 for cls in eval_class_ids}
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            # If masks are one-hot encoded, convert to class indices.
            if masks.dim() == 4:
                masks = masks.argmax(dim=1)  # shape: (B, H, W)
                
            seg_logits = model(images)  # shape: (B, 32, H, W)
            preds = seg_logits.argmax(dim=1)  # shape: (B, H, W), values in [0, 31]
            
            # Map the 32-class predictions to the 5 desired classes.
            mapped_preds = torch.zeros_like(preds)
            for train_cls, eval_cls in TRAIN_TO_EVAL_MAPPING.items():
                mapped_preds[preds == train_cls] = eval_cls
                
            # Compute intersection and union for each evaluation class.
            for cls in eval_class_ids:
                intersection = ((mapped_preds == cls) & (masks == cls)).sum().item()
                union = ((mapped_preds == cls) | (masks == cls)).sum().item()
                total_intersections[cls] += intersection
                total_unions[cls] += union
                
    # Calculate IoU per class and mean IoU.
    iou_per_class = np.array([
        total_intersections[cls] / (total_unions[cls] + 1e-6) 
        for cls in eval_class_ids
    ])
    mean_iou = iou_per_class.mean()
    
    return mean_iou, iou_per_class


def compute_average_precision(pred_scores, pred_matches, num_gt):
    """
    Compute average precision using the area under the precision-recall curve.
    This is a simplified approximation.
    """
    if len(pred_scores) == 0:
        return 0.0
    # Sort predictions by descending confidence
    sorted_indices = np.argsort(-np.array(pred_scores))
    sorted_matches = np.array(pred_matches)[sorted_indices]
    tp = sorted_matches
    fp = 1 - sorted_matches
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    precision = cum_tp / (cum_tp + cum_fp + 1e-6)
    recall = cum_tp / (num_gt + 1e-6)
    # Compute AP as area under the precision-recall curve (using trapezoidal rule)
    ap = np.trapz(precision, recall)
    return ap

def evaluate_detection(model, dataloader, device, iou_threshold=0.5):
    """
    Evaluate detection performance using a simplified mAP calculation.
    (Note: a full mAP implementation is more involved; this is a basic approximation.)
    
    Assumes:
      - The model detection head outputs boxes in format [xmax, xmin, ymax, ymin, confidence].
      - The dataset in bbox mode returns ground truth boxes in [x_min, y_min, x_max, y_max].
    """
    model.eval()
    pred_scores_all = []
    pred_matches_all = []
    total_gt = 0

    with torch.no_grad():
        for images, target in dataloader:
            images = images.to(device)
            # target is a dict with keys 'bboxes' and 'labels'
            bboxes_gt = target['bboxes']  # list or tensor of shape (B, num_boxes, 4)
            # Forward pass: we only use the detection branch here.
            _, det_preds = model(images)
            batch_size = images.size(0)
            for i in range(batch_size):
                # For each image:
                preds = det_preds[i]  # shape (num_anchors, 5)
                # Filter predictions by confidence threshold.
                confidence_threshold = model.confidence_threshold
                keep = preds[:, -1] > confidence_threshold
                preds = preds[keep]
                if preds.numel() == 0:
                    pred_boxes = np.empty((0, 4))
                    pred_confidences = np.empty((0,))
                else:
                    # Convert boxes from [xmax, xmin, ymax, ymin] to [x_min, y_min, x_max, y_max]
                    pred_boxes = torch.stack([preds[:,1], preds[:,3], preds[:,0], preds[:,2]], dim=1).cpu().numpy()
                    pred_confidences = preds[:, -1].cpu().numpy()
                # Ground truth boxes for image i (assumed already in [x_min, y_min, x_max, y_max])
                gt_boxes = bboxes_gt[i].cpu().numpy() if bboxes_gt[i].numel() > 0 else np.empty((0,4))
                total_gt += len(gt_boxes)
                # For each prediction, determine if it is a true positive.
                image_pred_matches = np.zeros(len(pred_boxes))
                gt_matched = np.zeros(len(gt_boxes))
                for j, pred_box in enumerate(pred_boxes):
                    best_iou = 0
                    best_idx = -1
                    for k, gt_box in enumerate(gt_boxes):
                        iou = compute_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = k
                    if best_iou >= iou_threshold and best_idx != -1 and gt_matched[best_idx] == 0:
                        image_pred_matches[j] = 1  # true positive
                        gt_matched[best_idx] = 1
                    else:
                        image_pred_matches[j] = 0  # false positive
                pred_scores_all.extend(pred_confidences.tolist())
                pred_matches_all.extend(image_pred_matches.tolist())

    ap = compute_average_precision(pred_scores_all, pred_matches_all, total_gt)
    return ap

def find_rare_classes_by_average(annotations_json):
    """
    Reads bounding box annotations, computes the average count of boxes 
    across the 5 target classes, and finds which classes are below average 
    along with how far below.

    Args:
        annotations_json (str): Path to the JSON file (e.g., train_bbox.json).

    Returns:
        below_avg_diff (dict): A dictionary of the form {class_id: diff_amount}, 
                               where diff_amount = (average - class_count) 
                               if class_count < average, else 0.
        class_counts (dict): {class_id: total number of bounding boxes} for each of the 5 classes.
        average_count (float): The average bounding box count across the 5 classes.
    """
    # Initialize counts for the 5 classes
    class_counts = {c: 0 for c in TARGET_CLASSES}

    # Load annotation data
    with open(annotations_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Accumulate bounding box counts
    for entry in data:
        bboxes = entry.get("bboxes", [])
        for bbox in bboxes:
            cls_id = bbox["class_id"]
            if cls_id in class_counts:
                class_counts[cls_id] += 1

    # Compute average box count across the 5 classes
    total_boxes = sum(class_counts.values())
    num_classes = len(TARGET_CLASSES)
    average_count = total_boxes / num_classes  # float

    # Identify classes below average and how far below
    below_avg_diff = {}
    for cls_id, count in class_counts.items():
        if count < average_count:
            diff = average_count - count  # how many boxes below average
            below_avg_diff[cls_id] = diff
        else:
            below_avg_diff[cls_id] = 0  # not below average

    return below_avg_diff, class_counts, average_count

def find_images_with_classes(annotations_json, target_class_ids):
    """
    Identify which images contain any bounding boxes of the given target_class_ids.

    Args:
        annotations_json (str): Path to the JSON file with bounding box annotations.
        target_class_ids (list of int): A list of class_ids (e.g. [4, 5] for motorcycle/bus).

    Returns:
        images_with_rare (list of str): Filenames of images that have at least one bbox 
                                        belonging to the target_class_ids.
    """
    images_with_rare = []

    with open(annotations_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Scan each image annotation for bounding boxes
    for entry in data:
        file_name = entry["file_name"]
        bboxes = entry.get("bboxes", [])
        found = False
        for bbox in bboxes:
            cls_id = bbox["class_id"]
            if cls_id in target_class_ids:
                found = True
                break
        if found:
            images_with_rare.append(file_name)

    return images_with_rare
