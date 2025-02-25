from src.utils import generate_annotations
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate bounding box annotations from segmentation masks.")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Path to the directory containing input images.")
    parser.add_argument("--masks_dir", type=str, required=True,
                        help="Path to the directory containing mask images.")
    parser.add_argument("--output_json", type=str, default="annotations.json",
                        help="Output JSON file path (default: annotations.json).")
    
    args = parser.parse_args()
    
    # Call the function
    generate_annotations(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        output_json=args.output_json
    )
    
if __name__ == "__main__":
    main()


"""
python annotate.py \
    --images_dir /path/to/images \
    --masks_dir /path/to/masks \
    --output_json /path/to/annotations.json

"""