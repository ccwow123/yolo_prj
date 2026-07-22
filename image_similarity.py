import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

def orb_distance(image1, image2):
    orb = cv2.ORB_create(nfeatures=500)
    
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None:
        return 1.0
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    if len(matches) == 0:
        return 1.0
    
    matches = sorted(matches, key=lambda x: x.distance)
    
    good_matches = [m for m in matches if m.distance < 50]
    
    if max(len(kp1), len(kp2)) == 0:
        return 1.0
    
    similarity = len(good_matches) / max(len(kp1), len(kp2))
    
    return 1.0 - similarity

def compare_images(img_path1, img_path2, threshold=0.7):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    
    if img1 is None:
        print(f"Error: Cannot read image {img_path1}")
        return None
    if img2 is None:
        print(f"Error: Cannot read image {img_path2}")
        return None
    
    distance = orb_distance(img1, img2)
    similarity = 100.0 * (1 - distance)
    
    return {
        'distance': distance,
        'similarity': similarity,
        'is_similar': distance <= threshold
    }

def filter_unique_images(input_dir, output_dir, threshold=0.7):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    files = sorted([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(image_extensions)])
    
    print(f"Found {len(files)} images in input directory")
    
    if len(files) == 0:
        print("No images found")
        return [], []
    
    images = {}
    for filename in tqdm(files, desc="Loading images", unit="image"):
        filepath = os.path.join(input_dir, filename)
        img = cv2.imread(filepath)
        if img is not None:
            images[filename] = img
    
    unique_images = []
    removed_images = []
    
    if files:
        unique_images.append(files[0])
        last_kept = files[0]
        
        for filename in tqdm(files[1:], desc="Comparing images", unit="image"):
            if last_kept not in images or filename not in images:
                unique_images.append(filename)
                last_kept = filename
                continue
                
            distance = orb_distance(images[last_kept], images[filename])
            
            if distance <= threshold:
                removed_images.append(filename)
                # print(f"  Remove {filename} (similar to {last_kept}, distance={distance:.4f})")
            else:
                unique_images.append(filename)
                last_kept = filename
                # print(f"  Keep {filename} (different from {last_kept}, distance={distance:.4f})")
    
    os.makedirs(output_dir, exist_ok=True)
    
    saved_count = 0
    for filename in tqdm(unique_images, desc="Saving unique images", unit="image"):
        src_path = os.path.join(input_dir, filename)
        dst_path = os.path.join(output_dir, filename)
        
        if os.path.exists(src_path):
            img = cv2.imread(src_path)
            if img is not None:
                cv2.imwrite(dst_path, img)
                saved_count += 1
    
    txt_path = os.path.join(output_dir, 'filter_report.txt')
    with open(txt_path, 'w') as f:
        f.write(f"Algorithm: ORB\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Total images processed: {len(files)}\n")
        f.write(f"Unique images saved: {len(unique_images)}\n")
        f.write(f"Images removed: {len(removed_images)}\n")
        
        if removed_images:
            f.write("\nRemoved images (similar to previous):\n")
            for filename in removed_images:
                f.write(f"  {filename}\n")
        
        f.write("\nKept images:\n")
        for filename in unique_images:
            f.write(f"  {filename}\n")
    
    print(f"\n=== Filter Complete ===")
    print(f"Algorithm: ORB")
    print(f"Threshold: {threshold}")
    print(f"Total images processed: {len(files)}")
    print(f"Unique images saved: {len(unique_images)}")
    print(f"Images removed: {len(removed_images)}")
    print(f"Results saved to: {output_dir}")
    
    return unique_images, removed_images

def run_similarity_check(args):
    if args.compare:
        if len(args.compare) != 2:
            print("Please provide exactly 2 images for comparison")
            return
        
        result = compare_images(args.compare[0], args.compare[1], args.threshold)
        if result:
            print(f"\nComparison Result:")
            print(f"Image 1: {args.compare[0]}")
            print(f"Image 2: {args.compare[1]}")
            print(f"Distance: {result['distance']:.4f}")
            print(f"Similarity: {result['similarity']:.2f}%")
            print(f"Is Similar: {result['is_similar']}")
    
    elif args.filter_unique:
        if not args.output:
            print("Please specify --output directory for filtered images")
            return
        
        filter_unique_images(args.filter_unique, args.output, args.threshold)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Similarity Detection using ORB Algorithm')
    
    parser.add_argument('--compare', nargs=2, metavar=('IMAGE1', 'IMAGE2'),
                        help='Compare two images for similarity')
    
    parser.add_argument('--filter-unique', type=str, default=r'runs\hand_distance\exp3\screenshots',
                        help='Filter and save only unique images to output directory')
    
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Distance threshold for similarity (0.0=identical, 1.0=completely different)')
    
    parser.add_argument('--output', type=str, default=r'runs\hand_distance\exp3\filtered',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    if not (args.compare or args.filter_unique):
        parser.print_help()
        print("\nPlease specify one of: --compare or --filter-unique")
    else:
        run_similarity_check(args)