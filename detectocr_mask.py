import cv2
import numpy as np
import easyocr
import time
import re
import os
from typing import List, Dict, Tuple, Optional
import json
from PIL import Image
from thefuzz import fuzz  # <-- ADD THIS IMPORT

def generate_mask_from_bbox(image_shape: Tuple[int, int], bbox_dict: Dict, padding: int = 10) -> np.ndarray:
    """
    Generates a binary mask image from a bounding box dictionary.

    Args:
        image_shape (Tuple[int, int]): The (height, width) of the original image.
        bbox_dict (Dict): Dictionary containing 'x_min', 'y_min', 'x_max', 'y_max'.
        padding (int): Pixels to add around the bounding box for the mask.

    Returns:
        np.ndarray: A binary mask (uint8) with the bbox area as white (255).
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # Apply padding, ensuring coordinates stay within image bounds
    x_min = max(0, bbox_dict["x_min"] - padding)
    y_min = max(0, bbox_dict["y_min"] - padding)
    x_max = min(width, bbox_dict["x_max"] + padding)
    y_max = min(height, bbox_dict["y_max"] + padding)

    # Draw a filled white rectangle on the mask
    cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)

    return mask


class ProductionOpenSooqDetector:
    def __init__(self, gpu: bool = False):
        """Initialize OCR reader and detection patterns"""
        self.gpu = gpu
        self.easyocr_reader = None
        self._initialize_ocr()
        self.arabic_patterns = ["السوق", "المفتوح", "السوقالمفتوح", "السوق المفتوح"]
        self.english_patterns = ["opensooq", "open", "sooq", "com", "opensooq.com"]

    def _initialize_ocr(self):
        """Initialize EasyOCR reader"""
        print("🔧 Initializing OCR...")
        # Ensure model directory exists or can be created
        model_dir = os.path.expanduser("~/.EasyOCR/model")
        os.makedirs(model_dir, exist_ok=True)
        self.easyocr_reader = easyocr.Reader(["en", "ar"], gpu=self.gpu, verbose=False, model_storage_directory=model_dir)
        print("✅ OCR ready")

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale and apply gamma correction"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Slightly adjusted gamma for potentially better contrast on watermarks
        enhanced = np.power(gray / 255.0, 0.5) * 255
        return enhanced.astype(np.uint8)

    # vvvvvv THIS METHOD IS UPDATED vvvvvv
    def _is_opensooq_text(self, text: str) -> Tuple[bool, float]:
        """Check if the given text contains OpenSooq watermark indicators using fuzzy matching."""
        text_lower = text.lower().strip()
        score = 0.0
        fuzzy_threshold = 80  # Similarity threshold (e.g., 80 out of 100)

        # --- Fuzzy Matching for Arabic Patterns ---
        # Find the best match ratio against all Arabic patterns
        max_arabic_ratio = 0
        if text:  # Ensure text is not empty
            max_arabic_ratio = max((fuzz.partial_ratio(p, text) for p in self.arabic_patterns), default=0)

        if max_arabic_ratio > fuzzy_threshold:
            # The score is proportional to how good the fuzzy match is
            score += (max_arabic_ratio / 100.0) * 3.0

        # --- Fuzzy Matching for English Patterns ---
        # Find the best match ratio against all English patterns
        max_english_ratio = 0
        if text_lower:  # Ensure text is not empty
            max_english_ratio = max((fuzz.partial_ratio(p, text_lower) for p in self.english_patterns), default=0)

        if max_english_ratio > fuzzy_threshold:
            score += (max_english_ratio / 100.0) * 2.0

        # --- Boost score for highly specific or exact matches (retains original logic's strength) ---
        if re.search(r"opensooq\.com", text_lower):
            score += 3.0  # Add a high bonus for the full domain

        # Bonus if both languages are likely present based on fuzzy matching
        if max_arabic_ratio > fuzzy_threshold and max_english_ratio > fuzzy_threshold:
            score += 5.0

        # Adjust the detection threshold based on the new, more sensitive scoring logic
        is_detected = score > 2.0
        return is_detected, score
    # ^^^^^^ THIS METHOD IS UPDATED ^^^^^^

    def detect_single_image(self, image_path: str, return_mask: bool = False, mask_padding: int = 10) -> Dict:
        """
        Detects OpenSooq watermark in a single image and optionally returns a mask.

        Args:
            image_path (str): Path to the image file.
            return_mask (bool): If True, generates and returns a binary mask.
            mask_padding (int): Padding around the detected bbox for the mask.

        Returns:
            Dict: Detection results, including 'mask' (np.ndarray) if return_mask is True.
        """
        try:
            # Use Pillow to open image first to handle potential format issues
            pil_image = Image.open(image_path).convert("RGB")
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            if image is None:
                return {"detected": False, "error": f"Cannot load image: {image_path}"}

            start_time = time.time()
            processed = self._preprocess_image(image)
            height, width = image.shape[:2]

            # Use paragraph=True might group text better, but let's stick to False for now
            results = self.easyocr_reader.readtext(processed, width_ths=0.8, height_ths=0.6, paragraph=False, batch_size=4)

            best_match, best_score = None, 0.0
            detected_watermarks = []

            for bbox, text, confidence in results:
                if confidence < 0.15: # Slightly lower confidence threshold
                    continue
                is_os, score = self._is_opensooq_text(text)
                print(f"OCR result:   '{text}' \t(Conf: {confidence:.2f}, OS Score: {score:.1f})")
                if is_os:
                    final_score = (confidence * 1.5) + score # Adjusted weighting
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    match_data = {
                        "text": text,
                        "confidence": float(confidence),
                        "opensooq_score": score,
                        "final_score": final_score,
                        "bbox": {
                            "x_min": int(min(x_coords)),
                            "y_min": int(min(y_coords)),
                            "x_max": int(max(x_coords)),
                            "y_max": int(max(y_coords)),
                        }
                    }
                    bbox_data = match_data["bbox"]
                    bbox_data["width"] = bbox_data["x_max"] - bbox_data["x_min"]
                    bbox_data["height"] = bbox_data["y_max"] - bbox_data["y_min"]
                    detected_watermarks.append(match_data)

            # Select the best overall match if multiple detected
            if detected_watermarks:
                best_match = max(detected_watermarks, key=lambda x: x["final_score"])
                best_score = best_match["final_score"]

            elapsed = time.time() - start_time
            output = {
                "detected": False,
                "processing_time": round(elapsed, 2),
                "image_size": {"width": width, "height": height}
            }

            if best_match:
                output["detected"] = True
                output["watermark"] = best_match
                print(f"   Best Match:  '{best_match['text']}' \t(Final Score: {best_score:.2f})")
                if return_mask:
                    output["mask"] = generate_mask_from_bbox((height, width), best_match["bbox"], padding=mask_padding)
            # else:
                # print("   No OpenSooq watermark detected.")

            return output

        except Exception as e:
            import traceback
            print(f"Error processing {image_path}: {e}")
            # traceback.print_exc()
            return {"detected": False, "error": str(e)}

    def detect_batch(self, image_paths: List[str], output_file: Optional[str] = None, return_masks: bool = False, mask_padding: int = 10) -> List[Dict]:
        """
        Detects watermarks in a batch of images.

        Args:
            image_paths (List[str]): List of image file paths.
            output_file (Optional[str]): Path to save JSON/CSV results.
            return_masks (bool): If True, include generated masks in the results.
            mask_padding (int): Padding for generated masks.

        Returns:
            List[Dict]: List of detection result dictionaries.
        """
        print(f"🚀 Processing {len(image_paths)} images...")
        results = []
        start = time.time()

        for idx, path in enumerate(image_paths, 1):
            print(f"📸 {idx}/{len(image_paths)}: {os.path.basename(path)}")
            result = self.detect_single_image(path, return_mask=return_masks, mask_padding=mask_padding)
            result["image_path"] = path
            result["image_name"] = os.path.basename(path)
            results.append(result)

            if result["detected"]:
                print(f"   ✅ Found: '{result['watermark']['text'][:50]}...' (score: {result['watermark']['final_score']:.1f})")
            elif 'error' in result:
                 print(f"   ❌ Error: {result['error']}")
            else:
                print(f"   ➖ Not found")

        elapsed = time.time() - start
        detected_count = sum(1 for r in results if r["detected"])
        error_count = sum(1 for r in results if 'error' in r and not r['detected'])

        print(f"\n📊 Done in {elapsed:.1f}s | Avg: {elapsed/len(image_paths):.1f}s/img")
        print(f"✅ Detected: {detected_count}/{len(image_paths)} ({(detected_count/len(image_paths))*100:.1f}%) | Errors: {error_count}")

        if output_file:
            # Exclude mask data from saved file for size reasons
            results_to_save = []
            for r in results:
                r_copy = r.copy()
                if "mask" in r_copy:
                    del r_copy["mask"]
                results_to_save.append(r_copy)
            self._save_results(results_to_save, output_file)
            print(f"💾 Saved detection results (without masks) to: {output_file}")

        return results

    def _save_results(self, results: List[Dict], output_file: str):
        # (Saving logic remains the same as provided)
        if output_file.endswith(".json"):
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        elif output_file.endswith(".csv"):
            try:
                import pandas as pd
                flat = []
                # Define all possible columns to ensure consistency
                cols = [
                    'image_path', 'image_name', 'detected', 'processing_time', 'error',
                    'text', 'confidence', 'opensooq_score', 'final_score',
                    'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'bbox_width', 'bbox_height'
                ]
                for r in results:
                    row = {k: r.get(k, '') for k in ['image_path', 'image_name', 'detected', 'processing_time', 'error']}
                    if r.get('detected'):
                        wm = r.get('watermark', {})
                        bbox = wm.get('bbox', {})
                        row.update({
                            'text': wm.get('text', ''),
                            'confidence': wm.get('confidence', ''),
                            'opensooq_score': wm.get('opensooq_score', ''),
                            'final_score': wm.get('final_score', ''),
                            'bbox_x_min': bbox.get('x_min', ''),
                            'bbox_y_min': bbox.get('y_min', ''),
                            'bbox_x_max': bbox.get('x_max', ''),
                            'bbox_y_max': bbox.get('y_max', ''),
                            'bbox_width': bbox.get('width', ''),
                            'bbox_height': bbox.get('height', ''),
                        })
                    # Ensure all columns exist, filling missing ones with empty string
                    for col in cols:
                        if col not in row:
                            row[col] = ''
                    flat.append(row)
                # Create DataFrame with specified column order
                df = pd.DataFrame(flat)[cols]
                df.to_csv(output_file, index=False, encoding='utf-8')
            except ImportError:
                print("Pandas not installed. Cannot save to CSV. Saving to JSON instead.")
                json_output_file = os.path.splitext(output_file)[0] + ".json"
                with open(json_output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"Saved results to: {json_output_file}")
            except Exception as e:
                 print(f"Error saving results to {output_file}: {e}")


# Production entrypoints (updated slightly)
def detect_opensooq_watermark(image_path: str, gpu: bool = False, return_mask: bool = False, mask_padding: int = 10) -> Dict:
    detector = ProductionOpenSooqDetector(gpu=gpu)
    return detector.detect_single_image(image_path, return_mask=return_mask, mask_padding=mask_padding)

def batch_detect_opensooq_watermarks(image_paths: List[str], output_file: Optional[str] = None, gpu: bool = False, return_masks: bool = False, mask_padding: int = 10) -> List[Dict]:
    detector = ProductionOpenSooqDetector(gpu=gpu)
    return detector.detect_batch(image_paths, output_file, return_masks=return_masks, mask_padding=mask_padding)

def detect_opensooq_in_folder(folder_path: str, output_file: Optional[str] = None,
                              extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
                              gpu: bool = False, return_masks: bool = False, mask_padding: int = 10) -> List[Dict]:
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if f.lower().endswith(extensions)]
    if not image_paths:
        print("❌ No images found.")
        return []
    print(f"📁 Found {len(image_paths)} images in {folder_path}")
    return batch_detect_opensooq_watermarks(image_paths, output_file, gpu, return_masks=return_masks, mask_padding=mask_padding)

# Example usage (demonstrating mask generation)
if __name__ == "__main__":
    print("🎯 OpenSooq Watermark Detector (Production Version with Mask Generation)")
    # Create a dummy image path for demonstration if needed
    # Example: test_image = "path/to/your/test_image.jpg"
    test_image = "/home/ubuntu/upload/op4.jpg" # Using the uploaded image
    output_mask_path = "/home/ubuntu/generated_mask.png"
    output_detection_results = "/home/ubuntu/detection_results.json"

    if os.path.exists(test_image):
        print(f"\n🔍 Testing single image detection with mask generation: {test_image}")
        # Set gpu=False explicitly if unsure about GPU availability/setup
        detection_result = detect_opensooq_watermark(test_image, gpu=False, return_mask=True, mask_padding=15)

        print("\n--- Detection Result ---")
        # Print result excluding the mask array itself for brevity
        result_summary = {k: v for k, v in detection_result.items() if k != 'mask'}
        print(json.dumps(result_summary, indent=2, ensure_ascii=False))
        print("------------------------")

        if detection_result.get("detected") and "mask" in detection_result:
            print(f"💾 Saving generated mask to: {output_mask_path}")
            mask_image = Image.fromarray(detection_result["mask"])
            mask_image.save(output_mask_path)
            print("✅ Mask saved.")
        elif 'error' in detection_result:
            print(f"⚠️ Detection failed: {detection_result['error']}")
        else:
            print("⚠️ Watermark not detected, mask not generated.")

        # Example for batch processing a folder (if you create a folder with images)
        # image_folder = "/home/ubuntu/test_images"
        # if os.path.exists(image_folder):
        #     print(f"\n🔄 Testing batch detection on folder: {image_folder}")
        #     batch_results = detect_opensooq_in_folder(image_folder, output_file=output_detection_results, gpu=False, return_masks=False)
        #     # Masks are not returned in batch by default to save memory, but results are saved to JSON/CSV.
        # else:
        #     print(f"\nFolder {image_folder} not found, skipping batch test.")

    else:
        print(f"❌ Test image not found: {test_image}")

