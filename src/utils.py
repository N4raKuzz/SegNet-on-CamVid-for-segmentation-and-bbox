import os
import cv2
import json
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 2 and h > 2:  # Ignore small noise
            bboxes.append({
                "class_id": int(class_id),
                "x_min": int(x - w/2),
                "y_min": int(y - h/2),
                "x_max": int(x + w/2),
                "y_max": int(y + h/2)
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
            
            # Get model outputs. If the model returns a tuple, take the first element.
            outputs = model(images)
            if isinstance(outputs, tuple):
                seg_logits = outputs[0]
            else:
                seg_logits = outputs  # shape: (B, 32, H, W)
            
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
    Evaluate detection performance using a simplified mAP calculation, and also compute 
    segmentation IoU if segmentation ground truth is available.
    
    Assumes:
      - The model detection head outputs boxes in format [xmax, xmin, ymax, ymin, confidence].
      - For detection, the dataset returns ground truth boxes in [x_min, y_min, x_max, y_max].
      - Optionally, the target dict may include a segmentation mask under key 'mask' 
        (with shape (H, W), class indices in [0, 31]).
    
    Returns:
        ap (float): Average precision computed for detection.
        mean_seg_iou (float): Mean segmentation IoU over evaluation classes.
        iou_per_class (np.array): IoU for each evaluation class.
    """
    model.eval()
    pred_scores_all = []
    pred_matches_all = []
    total_gt = 0

    # For segmentation evaluation, accumulate intersections and unions per evaluation class.
    # (Assumes TRAIN_TO_EVAL_MAPPING is defined, mapping from 32-class predictions to 5 eval classes.)
    eval_class_ids = sorted(set(TRAIN_TO_EVAL_MAPPING.values()))
    total_intersections_seg = {cls: 0 for cls in eval_class_ids}
    total_unions_seg = {cls: 0 for cls in eval_class_ids}
    
    with torch.no_grad():
        for images, target in dataloader:
            images = images.to(device)
            
            # Forward pass: get both segmentation and detection outputs.
            seg_logits, det_preds = model(images)
            
            # -------------------------
            # Detection evaluation
            # -------------------------
            # Expect target to be a dict with key 'bboxes' (and optionally 'mask').
            if isinstance(target, dict) and 'bboxes' in target:
                bboxes_gt = target['bboxes']  # list or tensor, one per image
            else:
                bboxes_gt = None

            batch_size = images.size(0)
            for i in range(batch_size):
                if bboxes_gt is not None:
                    preds = det_preds[i]  # shape (num_boxes, 5)
                    # Use the model's confidence threshold
                    confidence_threshold = model.confidence_threshold if hasattr(model, 'confidence_threshold') else 0.5
                    keep = preds[:, -1] > confidence_threshold
                    preds = preds[keep]
                    if preds.numel() == 0:
                        pred_boxes = np.empty((0, 4))
                        pred_confidences = np.empty((0,))
                    else:
                        # Convert from [xmax, xmin, ymax, ymin] to [x_min, y_min, x_max, y_max]
                        pred_boxes = torch.stack([preds[:, 1], preds[:, 3], preds[:, 0], preds[:, 2]], dim=1).cpu().numpy()
                        pred_confidences = preds[:, -1].cpu().numpy()
                    
                    gt_boxes = bboxes_gt[i].cpu().numpy() if bboxes_gt[i].numel() > 0 else np.empty((0, 4))
                    total_gt += len(gt_boxes)
                    
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
                    
                # -------------------------
                # Segmentation evaluation
                # -------------------------
                # If the target dict contains a segmentation mask under key 'mask'
                if isinstance(target, dict) and 'mask' in target:
                    # Assume segmentation ground truth mask is of shape (H, W) with class indices.
                    seg_gt = target['mask'][i].to(device)
                    # Get predicted segmentation from seg_logits (assumed shape: (B, 32, H, W))
                    pred_seg = seg_logits[i].argmax(dim=0)  # shape: (H, W)
                    
                    # Map 32-class predictions to evaluation classes.
                    mapped_preds = torch.zeros_like(pred_seg)
                    for train_cls, eval_cls in TRAIN_TO_EVAL_MAPPING.items():
                        mapped_preds[pred_seg == train_cls] = eval_cls
                    # Compute per-class intersection and union.
                    for cls in eval_class_ids:
                        intersection = ((mapped_preds == cls) & (seg_gt == cls)).sum().item()
                        union = ((mapped_preds == cls) | (seg_gt == cls)).sum().item()
                        total_intersections_seg[cls] += intersection
                        total_unions_seg[cls] += union

    # Compute Average Precision for detection.
    ap = compute_average_precision(pred_scores_all, pred_matches_all, total_gt) if total_gt > 0 else 0.0

    # Compute segmentation IoU per evaluation class and mean IoU.
    iou_per_class = np.array([
        total_intersections_seg[cls] / (total_unions_seg[cls] + 1e-6)
        for cls in eval_class_ids
    ])
    mean_seg_iou = iou_per_class.mean()
    
    return ap, mean_seg_iou, iou_per_class


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

def visualize_single_image_and_mask(image_path, mask_path, annotation_json_path):

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

    bboxes = annotation_for_image.get("bboxes", [])
    for bbox in bboxes:
        class_id = bbox["class_id"]
        x_min = bbox["x_min"]
        x_max = bbox["x_max"]
        y_min = bbox["y_min"]
        y_max = bbox["y_max"]
        
        color = TARGET_CLASS_COLORS.get(class_id, (255, 255, 255))
        thickness = 2
        
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        cv2.putText(image, f"ID:{class_id}", (x_min, max(y_min - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), color, thickness)
        cv2.putText(mask, f"ID:{class_id}", (x_min, max(y_min - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        print(f"Class ID: {class_id}, BBox => (x_min={x_min}, x_max={x_max}, "
              f"y_min={y_min}, y_max={y_max})")
        
    return image, mask

def colorize_mask(mask):
    """
    Convert a segmentation mask (with class labels) into a color image.
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in CLASS_COLOR_MAPPING.items():
        color_mask[mask == cls] = color
    return color_mask

def colorize_target_mask(mask):
    """
    Convert a target segmentation mask (with class ids) into a color image using TARGET_CLASS_COLORS.
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    # Invert the mapping: from class id to color.
    target_id_to_color = {class_id: color for color, class_id in TARGET_CLASS_COLORS.items()}
    for class_id, color in target_id_to_color.items():
        color_mask[mask == class_id] = color
    return color_mask

def visualize_predictions(model, dataset, device, num_samples=10):

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


def generate_proposals(seg_logits, seg_threshold=0.5, min_area=50):
    """
    Generate bounding box proposals from segmentation output.
    
    Args:
        seg_logits (torch.Tensor): Segmentation logits of shape (B, num_seg_classes, H, W).
                                   We assume a binary segmentation (2 channels) where channel 1 is foreground.
        seg_threshold (float): Threshold for foreground probability.
        min_area (int): Minimum area (in pixels) for a detected region to be considered.
        
    Returns:
        proposals (list of torch.Tensor): A list of length B. For each image, a tensor of shape (N, 5)
                                          where each row is [xmin, ymin, xmax, ymax, confidence].
                                          The coordinates are in the original image scale.
    """
    proposals_batch = []
    # Compute probabilities (assumes binary segmentation with two channels)
    probs = torch.softmax(seg_logits, dim=1)  # shape: (B, 2, H, W)
    # Use channel 1 as foreground probability.
    foreground_prob = probs[:, 1].detach().cpu().numpy()  # shape: (B, H, W)
    
    B = seg_logits.shape[0]
    for i in range(B):
        prob_map = foreground_prob[i]
        # Threshold to get a binary mask.
        mask = (prob_map > seg_threshold).astype(np.uint8)
        # Find contours (external only) using OpenCV.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        proposals = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                # Compute a confidence score as the average foreground probability in this region.
                region = prob_map[y:y+h, x:x+w]
                conf = float(np.mean(region))
                # Format: [xmin, ymin, xmax, ymax, confidence]
                proposals.append([x, y, x + w, y + h, conf])
        if proposals:
            proposals_tensor = torch.tensor(proposals, dtype=torch.float32, device=seg_logits.device)
        else:
            proposals_tensor = torch.empty((0, 5), dtype=torch.float32, device=seg_logits.device)
        proposals_batch.append(proposals_tensor)
    return proposals_batch

def roi_pooling(feature_map, boxes, output_size=(7, 7)):
    """
    A simple ROI pooling operation.
    
    Args:
        feature_map (torch.Tensor): Feature map of shape (C, H, W).
        boxes (torch.Tensor): Tensor of shape (N, 4) containing bounding boxes
                              in the coordinate space of feature_map (i.e. already scaled).
                              Each box is [xmin, ymin, xmax, ymax].
        output_size (tuple): Desired output size (height, width) for each ROI.
        
    Returns:
        rois (torch.Tensor): Tensor of shape (N, C, output_size[0], output_size[1]).
    """
    pooled_rois = []
    C, H, W = feature_map.shape
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        # Convert to integer indices.
        xmin = int(torch.floor(xmin).item())
        ymin = int(torch.floor(ymin).item())
        xmax = int(torch.ceil(xmax).item())
        ymax = int(torch.ceil(ymax).item())
        # Clamp coordinates to the feature map dimensions.
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(W, xmax)
        ymax = min(H, ymax)
        # If the region is empty, use a zeros tensor.
        if xmax <= xmin or ymax <= ymin:
            roi = torch.zeros((C, output_size[0], output_size[1]), device=feature_map.device)
        else:
            roi = feature_map[:, ymin:ymax, xmin:xmax]
            # Use adaptive average pooling to get a fixed output size.
            roi = F.adaptive_avg_pool2d(roi, output_size)
        pooled_rois.append(roi)
    if pooled_rois:
        return torch.stack(pooled_rois, dim=0)
    else:
        return torch.empty((0, C, output_size[0], output_size[1]), device=feature_map.device)

