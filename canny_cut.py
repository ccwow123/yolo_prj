import cv2
import numpy as np
import os
import argparse

# Global variables for interactive mode
img = None
window_name = "Comic Book Rectification"
crop_ratio = 0.2

params = {
    'blur': 5,
    'canny_low': 50,
    'canny_high': 150,
    'dilate': 0,
    'erode': 0,
    'epsilon_ratio': 2,
    'min_area': 5000,
    'max_area': 1000000
}

def crop_sides(image, crop_ratio=0.0):
    h, w = image.shape[:2]
    crop_width = int(w * crop_ratio / 2)
    if crop_width > 0:
        return image[:, crop_width:w-crop_width]
    return image

def detect_corners(image):
    image = cv2.GaussianBlur(image, (7, 7), 0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blur_kernel = params['blur'] * 2 + 1
    blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    edges = cv2.Canny(blur, params['canny_low'], params['canny_high'])
    
    if params['dilate'] > 0:
        kernel = np.ones((params['dilate'], params['dilate']), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    
    if params['erode'] > 0:
        kernel = np.ones((params['erode'], params['erode']), np.uint8)
        edges = cv2.erode(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, edges, []
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < params['min_area'] or area > params['max_area']:
            continue
        
        peri = cv2.arcLength(cnt, True)
        epsilon = params['epsilon_ratio'] / 100.0 * peri
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        if len(approx) == 4:
            return approx.reshape(4, 2), edges, contours
    
    return None, edges, contours

def rectify_book(image, corners):
    pts = np.array(corners, dtype=np.float32)
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    h, w = image.shape[:2]
    
    dst = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(image, M, (w, h))
    
    return warp

def update_param(key, value):
    params[key] = value
    update_display()

def update_display():
    if img is None:
        return
    
    cropped_img = crop_sides(img, crop_ratio)
    corners, edges, contours = detect_corners(cropped_img)
    
    vis = cropped_img.copy()
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    if contours:
        cv2.drawContours(vis, contours[0], -1, (0, 255, 0), 2)
    
    if corners is not None:
        for i, (x, y) in enumerate(corners):
            cv2.circle(vis, (int(x), int(y)), 8, (0, 0, 255), -1)
            cv2.putText(vis, str(i), (int(x) + 15, int(y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.polylines(vis, [corners.astype(np.int32)], True, (255, 0, 0), 2)
    
    h, w = vis.shape[:2]
    combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
    combined[:, :w] = edges_rgb
    combined[:, w:] = vis
    
    cv2.putText(combined, "Edge Detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Corner Detection", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    if corners is not None:
        cv2.putText(combined, "Detected!", (w + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(combined, "Not Found", (w + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.imshow(window_name, combined)

def init_trackbars():
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 400, 400)
    
    cv2.createTrackbar("Blur", "Controls", params['blur'], 10, 
                       lambda v: update_param('blur', v))
    cv2.setTrackbarMin("Blur", "Controls", 1)
    
    cv2.createTrackbar("Canny Low", "Controls", params['canny_low'], 255, 
                       lambda v: update_param('canny_low', v))
    
    cv2.createTrackbar("Canny High", "Controls", params['canny_high'], 255, 
                       lambda v: update_param('canny_high', v))
    
    cv2.createTrackbar("Dilate", "Controls", params['dilate'], 10, 
                       lambda v: update_param('dilate', v))
    
    cv2.createTrackbar("Erode", "Controls", params['erode'], 10, 
                       lambda v: update_param('erode', v))
    
    cv2.createTrackbar("Epsilon", "Controls", params['epsilon_ratio'], 10, 
                       lambda v: update_param('epsilon_ratio', v))
    cv2.setTrackbarMin("Epsilon", "Controls", 1)
    
    cv2.createTrackbar("Min Area", "Controls", params['min_area'] // 100, 200, 
                       lambda v: update_param('min_area', v * 100))
    
    cv2.createTrackbar("Max Area", "Controls", params['max_area'] // 10000, 100, 
                       lambda v: update_param('max_area', v * 10000))

def interactive_mode(image_path, output_path, ratio=0.2):
    global img, crop_ratio
    crop_ratio = ratio
    
    if not os.path.isfile(image_path):
        print(f"Error: File not found - {image_path}")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image - {image_path}")
        return
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 600)
    
    init_trackbars()
    update_display()
    
    print("=== Interactive Comic Book Rectification ===")
    print("Adjust sliders to detect 4 corners")
    print("Press 's' to save rectified image")
    print("Press 'q' to quit")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            cv2.destroyAllWindows()
            return
        elif key == ord('s'):
            cropped_img = crop_sides(img, crop_ratio)
            corners, _, _ = detect_corners(cropped_img)
            
            if corners is not None:
                warp = rectify_book(cropped_img, corners)
                
                save_path = output_path
                _, ext = os.path.splitext(save_path)
                if not ext.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'):
                    save_path = save_path + '.jpg'
                
                cv2.imwrite(save_path, warp)
                print(f"\nSaved to: {save_path}")
                print(f"Output size: {warp.shape[1]}x{warp.shape[0]}")
            else:
                print("\nFailed: No 4 corners detected")
    
    cv2.destroyAllWindows()

def batch_process(input_dir, output_dir, ratio=0.2):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    files = sorted([f for f in os.listdir(input_dir) 
                   if os.path.isfile(os.path.join(input_dir, f)) 
                   and f.lower().endswith(image_extensions)])
    
    if not files:
        print("Error: No images found in input directory")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {len(files)} images (crop ratio: {ratio})...")
    
    for filename in files:
        input_path = os.path.join(input_dir, filename)
        name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{name}_rect.jpg")
        
        image = cv2.imread(input_path)
        if image is None:
            print(f"  ✗ {filename} - Cannot read")
            continue
        
        cropped_img = crop_sides(image, ratio)
        corners, _, _ = detect_corners(cropped_img)
        
        if corners is not None:
            warp = rectify_book(cropped_img, corners)
            cv2.imwrite(output_path, warp)
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} - No corners")

def main():
    parser = argparse.ArgumentParser(description='Comic Book Page Rectification')
    parser.add_argument('--input', type=str, default = r'C:\Users\Administrator\Desktop\1.png', help='Input image or directory')
    parser.add_argument('--output', type=str, default='./runs/output', help='Output path or directory')
    parser.add_argument('--tune', default='True', help='Interactive parameter tuning')
    parser.add_argument('--crop', type=float, default=0, help='Side crop ratio (0.0 = no crop, 0.5 = crop half)')
    
    args = parser.parse_args()
    
    if args.tune:
        if os.path.isdir(args.input):
            files = [f for f in os.listdir(args.input) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if files:
                interactive_mode(os.path.join(args.input, files[0]), args.output, args.crop)
            else:
                print("No images found in input directory")
        else:
            interactive_mode(args.input, args.output, args.crop)
        return
    
    if os.path.isdir(args.input):
        batch_process(args.input, args.output, args.crop)
    else:
        image = cv2.imread(args.input)
        if image is None:
            print(f"Error: Cannot read image - {args.input}")
            return
        
        cropped_img = crop_sides(image, args.crop)
        corners, _, _ = detect_corners(cropped_img)
        
        if corners is not None:
            warp = rectify_book(cropped_img, corners)
            
            output_path = args.output
            _, ext = os.path.splitext(output_path)
            if not ext.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'):
                output_path = output_path + '.jpg'
            
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(output_path, warp)
            print(f"Successfully saved to: {output_path}")
            print(f"Output size: {warp.shape[1]}x{warp.shape[0]}")
        else:
            print("Failed: No 4 corners detected. Try using --tune to adjust parameters.")

if __name__ == "__main__":
    main()