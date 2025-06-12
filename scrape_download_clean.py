#!/usr/bin/env python3

import requests
import time
import datetime
import os
import sys
import argparse
import json
import hashlib
import sqlite3
from urllib.parse import urljoin, urlparse, parse_qs, urlencode
from typing import List, Dict, Optional, Tuple

# Ensure the directory containing the parser and detector is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

try:
    from opensooq_parser import OpenSooqScraper
except ImportError as e:
    print(f"Error importing OpenSooqScraper: {e}")
    print("Please ensure opensooq_parser.py is in the same directory or Python path.")
    sys.exit(1)

try:
    from detectocr_mask import ProductionOpenSooqDetector
    from PIL import Image, UnidentifiedImageError
except ImportError as e:
    print(f"Error importing ProductionOpenSooqDetector or PIL: {e}")
    print("Please ensure detectocr_mask.py is in the same directory and Pillow is installed.")
    sys.exit(1)

try:
    from simple_lama_inpainting import SimpleLama
except ImportError as e:
    print(f"Error importing SimpleLama: {e}")
    print("Please ensure simple-lama-inpainting is installed (`pip install simple-lama-inpainting`)")
    sys.exit(1)

# --- Configuration ---
BASE_URL = "https://jo.opensooq.com"
START_CATEGORY_URL = "https://jo.opensooq.com/ar/property/property-for-sale"
MAX_PAGES_TO_SCRAPE = 2
REQUEST_DELAY_SECONDS = 0 # Be polite to the server
REQUEST_TIMEOUT_SECONDS = 60 # Increased timeout for larger images
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
DOWNLOAD_DIR = "downloaded_images_jpg_v3"
CLEANED_DIR = "cleaned_images_jpg_v3"
DOWNLOAD_LOG = "download_log_v3.json"
WATERMARK_LOG = "watermark_removal_log_v3.json"
SCRAPED_DATA_JSON = "scraped_listings_v3.json"
TRACKING_DB = "scraped_listings_v3.db"

# --- Database Setup ---
def setup_tracking_db(db_path: str):
    """Creates the tracking database and table if they don't exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scraped_posts (
            url TEXT PRIMARY KEY,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def check_if_scraped(url: str, db_path: str) -> bool:
    """Checks if a listing URL has already been scraped."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM scraped_posts WHERE url = ?", (url,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def mark_as_scraped(url: str, db_path: str):
    """Marks a listing URL as scraped in the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT OR IGNORE INTO scraped_posts (url) VALUES (?)", (url,))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error marking {url} as scraped: {e}")
    finally:
        conn.close()

# --- Helper Functions ---

def fetch_html(url: str) -> str | None:
    """Fetches HTML content from a URL with error handling."""
    headers = {"User-Agent": USER_AGENT}
    print(f"Fetching HTML: {url}")
    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        print(f"  Status Code: {response.status_code}")
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching {url}: {e}")
        return None
    except Exception as e:
        print(f"  An unexpected error occurred during fetch: {e}")
        return None

def scrape_multiple_pages(start_url: str, max_pages: int, db_path: str, date_limit_days: Optional[int] = None, sort_recent: bool = True) -> list:
    """Scrapes listings from multiple pages, respecting date limits and avoiding duplicates."""
    all_listings = []
    current_url = OpenSooqScraper.construct_search_url(start_url, page=1, sort_recent=sort_recent)
    pages_scraped = 0
    stop_scraping_due_to_date = False
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    date_threshold = now_utc - datetime.timedelta(days=date_limit_days) if date_limit_days is not None else None

    while pages_scraped < max_pages and current_url and not stop_scraping_due_to_date:
        print(f"\n--- Scraping Page {pages_scraped + 1} --- ")
        fetch_time = datetime.datetime.now(datetime.timezone.utc)
        html_content = fetch_html(current_url)

        if not html_content:
            print("Failed to fetch page content. Stopping scrape.")
            break

        try:
            parser = OpenSooqScraper(html_content, base_url=BASE_URL)
            listings_on_page = parser.parse_listings(fetch_time)
            print(f"Parsed {len(listings_on_page)} listings from this page.")
            new_listings_count = 0
            for listing in listings_on_page:
                listing_url = listing.get("url")
                if not listing_url:
                    continue

                if check_if_scraped(listing_url, db_path):
                    print(f"  Skipping already scraped: {listing_url}")
                    continue

                posted_dt_str = listing.get("posted_datetime")
                if date_threshold and posted_dt_str:
                    try:
                        posted_dt = datetime.datetime.fromisoformat(posted_dt_str)
                        if posted_dt.tzinfo is None:
                            posted_dt = posted_dt.replace(tzinfo=datetime.timezone.utc)
                        if posted_dt < date_threshold:
                            print(f"  Listing {listing_url} (posted: {posted_dt_str}) is older than threshold ({date_limit_days} days). Skipping.")
                            stop_scraping_due_to_date = True
                            continue
                    except ValueError:
                        print(f"  Invalid datetime format: {posted_dt_str}")
                        continue

                # --- Fetch listing details ---
                print(f"  Fetching details for: {listing_url}")
                detail_html = fetch_html(listing_url)
                if not detail_html:
                    print(f"    Failed to fetch detail page: {listing_url}")
                    continue
                # time.sleep(0.5) # Small delay before parsing details

                detail_parser = OpenSooqScraper(detail_html, base_url=BASE_URL)
                detail_info = detail_parser.parse_listing_details(fetch_time)
                print(f"    Parsed details, found {len(detail_info.get('image_urls', []))} image URLs.")

                # Merge summary + detail info
                merged = {**listing, **detail_info}

                all_listings.append(merged)
                mark_as_scraped(listing_url, db_path)
                new_listings_count += 1

            print(f"Added {new_listings_count} new listings from this page.")

            if stop_scraping_due_to_date:
                 print("Stopping pagination due to reaching date threshold.")
                 break

            pagination_info = parser.get_pagination_info()
            next_page_relative_url = pagination_info.get("next_page_url")
            if next_page_relative_url:
                current_url = urljoin(BASE_URL, next_page_relative_url)
                if sort_recent:
                    parsed_next = urlparse(current_url)
                    query_params = parse_qs(parsed_next.query)
                    if 'sort_code' not in query_params:
                        query_params['sort_code'] = ['recent']
                        query_params['search'] = ['true']
                        new_query = urlencode(query_params, doseq=True)
                        current_url = parsed_next._replace(query=new_query).geturl()
            else:
                current_url = None

            pages_scraped += 1

            if not current_url:
                print("No next page found.")
                break
            elif pages_scraped < max_pages:
                 print(f"Next page URL: {current_url}")
                 print(f"Waiting {REQUEST_DELAY_SECONDS} seconds before next request...")
                 time.sleep(REQUEST_DELAY_SECONDS)
            else:
                print("Reached max pages limit.")
                break

        except Exception as e:
            import traceback
            print(f"Error parsing page {current_url}: {e}")
            traceback.print_exc()
            break

    print(f"\n--- Scraping Finished --- ")
    print(f"Total pages checked: {pages_scraped}")
    print(f"Total new listings collected: {len(all_listings)}")
    return all_listings

def download_image_as_jpg(url: str, save_path_jpg: str) -> Tuple[bool, str]:
    """Downloads an image from a URL and saves it as JPG."""
    if not url or not url.startswith("http"):
        return False, "Invalid URL"
    headers = {"User-Agent": USER_AGENT}
    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS, stream=True)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if content_type and not content_type.startswith("image/"):
            print(f"    Warning: Content-Type ({content_type}) doesn't look like an image, attempting conversion anyway.")
        try:
            img = Image.open(response.raw)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.save(save_path_jpg, "JPEG", quality=95)
            return True, "Success"
        except UnidentifiedImageError:
            return False, "Cannot identify image file (PIL)"
        except IOError as e:
            return False, f"PIL file error: {e}"
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else None
        return False, f"Request error: {e} (Status: {status_code})"
    except Exception as e:
        return False, f"Unexpected error during download/conversion: {e}"

def generate_filename_from_url_jpg(url: str, listing_url_hash: str, image_index: int) -> str:
    """Generates a unique filename with .jpg extension from a URL, incorporating listing hash and index."""
    try:
        parsed_url = urlparse(url)
        path_query = parsed_url.path + ("?" + parsed_url.query if parsed_url.query else "")
        image_hash = hashlib.sha1(path_query.encode("utf-8")).hexdigest()[:8]
        return f"{listing_url_hash}_{image_index:02d}_{image_hash}.jpg"
    except Exception:
        fallback_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
        return f"{listing_url_hash}_{image_index:02d}_{fallback_hash}.jpg"

def download_images_for_listings(listings: List[Dict], download_dir: str) -> List[Dict]:
    """Downloads ALL images listed in 'image_urls' for each listing as JPG."""
    os.makedirs(download_dir, exist_ok=True)
    download_log = []
    total_images_attempted = 0
    total_images_succeeded = 0

    print(f"\n--- Starting Image Downloads (All Listing Images) (Output Dir: {download_dir}) ---")

    for i, listing in enumerate(listings):
        listing_url = listing.get("url", f"unknown_listing_{i}")
        listing_title = listing.get("title", "N/A")[:50]
        image_urls = listing.get("image_urls", [])

        # Generate a hash for the listing URL to use in filenames
        listing_url_hash = hashlib.sha1(listing_url.encode("utf-8")).hexdigest()[:10]

        listing['downloaded_paths'] = [] # Initialize list to store paths of successfully downloaded images

        if not image_urls:
            print(f"Listing {i+1}/{len(listings)} ({listing_title}): No image URLs found.")
            log_entry = {
                "listing_index": i,
                "listing_url": listing_url,
                "listing_title": listing_title,
                "image_index": -1,
                "image_url": None,
                "download_status": "Skipped",
                "message": "No image_urls field found or empty.",
                "local_path": None
            }
            download_log.append(log_entry)
            continue

        print(f"Listing {i+1}/{len(listings)} ({listing_title}): Found {len(image_urls)} images.")

        for img_idx, img_url in enumerate(image_urls):
            total_images_attempted += 1
            filename = generate_filename_from_url_jpg(img_url, listing_url_hash, img_idx)
            save_path = os.path.join(download_dir, filename)

            log_entry = {
                "listing_index": i,
                "listing_url": listing_url,
                "listing_title": listing_title,
                "image_index": img_idx,
                "image_url": img_url,
                "download_status": "Failed",
                "message": "",
                "local_path": save_path
            }

            print(f"  Downloading image {img_idx+1}/{len(image_urls)}: {img_url} -> {filename}")
            success, message = download_image_as_jpg(img_url, save_path)

            if success:
                print(f"    Success.")
                log_entry["download_status"] = "Success"
                log_entry["message"] = "Image downloaded successfully as JPG."
                listing['downloaded_paths'].append(save_path)
                total_images_succeeded += 1
            else:
                print(f"    Failed: {message}")
                log_entry["message"] = f"Download failed: {message}"
                # Clean up failed download file if it exists
                if os.path.exists(save_path):
                    try: os.remove(save_path)
                    except OSError as e: print(f"    Warning: Could not remove failed download file {save_path}: {e}")

            download_log.append(log_entry)
            # time.sleep(0.1) # Small delay between image downloads

    print(f"\n--- Image Downloads Finished --- ")
    print(f"Attempted: {total_images_attempted} images")
    print(f"Succeeded: {total_images_succeeded} images")
    print(f"Failed: {total_images_attempted - total_images_succeeded} images")

    try:
        with open(DOWNLOAD_LOG, "w", encoding="utf-8") as f:
            json.dump(download_log, f, indent=2, ensure_ascii=False)
        print(f"Download log saved to: {DOWNLOAD_LOG}")
    except IOError as e:
        print(f"Error saving download log: {e}")

    return listings

# --- Watermark Removal Function (MODIFIED to handle multiple images per listing) ---
def remove_watermarks_batch(listings_with_paths: List[Dict], cleaned_dir: str, detector: ProductionOpenSooqDetector, inpainter: SimpleLama, mask_padding: int = 15) -> List[Dict]:
    """Detects and removes watermarks from all downloaded images for each listing."""
    os.makedirs(cleaned_dir, exist_ok=True)
    watermark_log = []
    total_images_processed = 0
    total_watermarks_detected = 0
    total_watermarks_removed = 0
    total_errors = 0

    print(f"\n--- Starting Watermark Detection & Removal (Output Dir: {cleaned_dir}) ---")

    if not detector:
        print("Watermark detector not initialized. Skipping removal.")
        return listings_with_paths
    if not inpainter:
        print("Inpainter not initialized. Skipping removal.")
        return listings_with_paths

    for i, listing in enumerate(listings_with_paths):
        listing_url = listing.get("url", f"unknown_listing_{i}")
        listing_title = listing.get("title", "N/A")[:50]
        downloaded_paths = listing.get("downloaded_paths", [])

        listing['cleaned_paths'] = [] # Initialize list for cleaned image paths

        if not downloaded_paths:
            print(f"Listing {i+1}/{len(listings_with_paths)} ({listing_title}): No downloaded images to process.")
            continue

        print(f"Listing {i+1}/{len(listings_with_paths)} ({listing_title}): Processing {len(downloaded_paths)} images for watermarks.")

        for img_idx, downloaded_path in enumerate(downloaded_paths):
            total_images_processed += 1
            base_filename = os.path.basename(downloaded_path)
            filename_stem = os.path.splitext(base_filename)[0]
            cleaned_filename = f"{filename_stem}_cleaned.jpg"
            cleaned_path = os.path.join(cleaned_dir, cleaned_filename)

            log_entry = {
                "listing_index": i,
                "listing_url": listing_url,
                "listing_title": listing_title,
                "image_index": img_idx,
                "original_path": downloaded_path,
                "cleaned_path": None,
                "detection_status": "Not Detected",
                "removal_status": "N/A",
                "message": "",
                "detection_details": None
            }

            print(f"  Processing image {img_idx+1}/{len(downloaded_paths)}: {base_filename}")

            if not os.path.exists(downloaded_path):
                print(f"    Error: Original file not found: {downloaded_path}")
                log_entry["message"] = "Original file missing."
                log_entry["detection_status"] = "Error"
                total_errors += 1
                watermark_log.append(log_entry)
                continue

            try:
                # 1. Detect Watermark
                detection_result = detector.detect_single_image(downloaded_path, return_mask=True, mask_padding=mask_padding)
                log_entry["detection_details"] = {k: v for k, v in detection_result.items() if k != 'mask'} # Log details except mask array

                if detection_result.get("error"):
                    print(f"    Detection Error: {detection_result['error']}")
                    log_entry["message"] = f"Detection error: {detection_result['error']}"
                    log_entry["detection_status"] = "Error"
                    total_errors += 1

                elif detection_result.get("detected"):
                    print(f"    Watermark Detected: '{detection_result['watermark']['text'][:30]}...' (Score: {detection_result['watermark']['final_score']:.1f}) ")
                    log_entry["detection_status"] = "Detected"
                    total_watermarks_detected += 1

                    # 2. Remove Watermark (Inpaint)
                    mask = detection_result.get("mask")
                    if mask is not None:
                        try:
                            print(f"    Inpainting mask... -> {cleaned_filename}")
                            # Load image with PIL for inpainter
                            img_pil = Image.open(downloaded_path).convert('RGB')
                            mask_pil = Image.fromarray(mask).convert('L') # Ensure mask is grayscale

                            # Perform inpainting
                            inpainted_img_pil = inpainter(img_pil, mask_pil)

                            # Save the cleaned image
                            inpainted_img_pil.save(cleaned_path, "JPEG", quality=95)
                            print(f"    Inpainting successful.")
                            log_entry["removal_status"] = "Removed"
                            log_entry["cleaned_path"] = cleaned_path
                            listing['cleaned_paths'].append(cleaned_path)
                            total_watermarks_removed += 1
                        except Exception as inpaint_err:
                            print(f"    Inpainting Error: {inpaint_err}")
                            log_entry["message"] = f"Inpainting error: {inpaint_err}"
                            log_entry["removal_status"] = "Error"
                            total_errors += 1
                    else:
                        print("    Error: Detection successful but mask not generated.")
                        log_entry["message"] = "Mask not generated despite detection."
                        log_entry["removal_status"] = "Error"
                        total_errors += 1
                else:
                    print("    Watermark Not Detected.")
                    log_entry["detection_status"] = "Not Detected"
                    # Optionally copy the original image to cleaned dir if no watermark
                    # shutil.copy2(downloaded_path, cleaned_path)
                    # listing['cleaned_paths'].append(cleaned_path)
                    # log_entry["removal_status"] = "Not Applicable (No Watermark)"
                    # log_entry["cleaned_path"] = cleaned_path

            except Exception as process_err:
                import traceback
                print(f"    Unexpected Error processing image: {process_err}")
                # traceback.print_exc()
                log_entry["message"] = f"Unexpected processing error: {process_err}"
                log_entry["detection_status"] = "Error"
                log_entry["removal_status"] = "Error"
                total_errors += 1

            watermark_log.append(log_entry)
            # time.sleep(0.1) # Small delay between image processing

    print(f"\n--- Watermark Removal Finished --- ")
    print(f"Total images processed: {total_images_processed}")
    print(f"Watermarks detected: {total_watermarks_detected}")
    print(f"Watermarks successfully removed: {total_watermarks_removed}")
    print(f"Errors during processing: {total_errors}")

    try:
        with open(WATERMARK_LOG, "w", encoding="utf-8") as f:
            # Exclude mask data from saved log for size reasons
            log_to_save = []
            for entry in watermark_log:
                entry_copy = entry.copy()
                if entry_copy.get("detection_details") and "mask" in entry_copy["detection_details"]:
                     del entry_copy["detection_details"]["mask"]
                log_to_save.append(entry_copy)
            json.dump(log_to_save, f, indent=2, ensure_ascii=False)
        print(f"Watermark removal log saved to: {WATERMARK_LOG}")
    except IOError as e:
        print(f"Error saving watermark removal log: {e}")

    return listings_with_paths

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Scrape OpenSooq listings, download images, and remove watermarks.")
    parser.add_argument("--start-url", default=START_CATEGORY_URL, help="The starting category URL on OpenSooq.")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES_TO_SCRAPE, help="Maximum number of pages to scrape.")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY_SECONDS, help="Delay between requests in seconds.")
    parser.add_argument("--timeout", type=int, default=REQUEST_TIMEOUT_SECONDS, help="Request timeout in seconds.")
    parser.add_argument("--download-dir", default=DOWNLOAD_DIR, help="Directory to save downloaded images.")
    parser.add_argument("--cleaned-dir", default=CLEANED_DIR, help="Directory to save cleaned images.")
    parser.add_argument("--db-path", default=TRACKING_DB, help="Path to the SQLite tracking database.")
    parser.add_argument("--output-json", default=SCRAPED_DATA_JSON, help="Path to save the final scraped data JSON file.")
    parser.add_argument("--date-limit-days", type=int, default=None, help="Only scrape listings posted within the last N days.")
    parser.add_argument("--skip-download", action="store_true", help="Skip the image downloading step.")
    parser.add_argument("--skip-cleaning", action="store_true", help="Skip the watermark removal step.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for OCR and Inpainting if available.")
    parser.add_argument("--mask-padding", type=int, default=15, help="Padding around detected watermark for mask generation.")

    args = parser.parse_args()


    print("--- Starting OpenSooq Scraper V3 --- ")
    print(f"Config: Max Pages={args.max_pages}, Delay={args.delay}s, Date Limit={args.date_limit_days} days")
    print(f"DB: {args.db_path}, Download Dir: {args.download_dir}, Cleaned Dir: {args.cleaned_dir}")
    print(f"Output JSON: {args.output_json}")
    print(f"GPU Enabled: {args.gpu}")

    # Setup
    setup_tracking_db(args.db_path)
    os.makedirs(args.download_dir, exist_ok=True)
    os.makedirs(args.cleaned_dir, exist_ok=True)

    # 1. Scrape Listings (includes fetching details)
    scraped_listings = scrape_multiple_pages(
        start_url=args.start_url,
        max_pages=args.max_pages,
        db_path=args.db_path,
        date_limit_days=args.date_limit_days,
        sort_recent=True # Defaulting to sort by recent
    )

    if not scraped_listings:
        print("No new listings were scraped. Exiting.")
        return

    # 2. Download Images
    listings_with_downloads = []
    if not args.skip_download:
        listings_with_downloads = download_images_for_listings(
            listings=scraped_listings,
            download_dir=args.download_dir
        )
    else:
        print("Skipping image download step as requested.")
        listings_with_downloads = scraped_listings # Pass through if skipping
        # Ensure 'downloaded_paths' exists even if empty when skipping
        for listing in listings_with_downloads:
            if 'downloaded_paths' not in listing:
                listing['downloaded_paths'] = []

    # 3. Remove Watermarks
    listings_final = []
    if not args.skip_cleaning:
        print("Initializing Watermark Detector and Inpainter...")
        try:
            detector = ProductionOpenSooqDetector(gpu=args.gpu)
        except Exception as e:
            print(f"Failed to initialize OCR Detector: {e}. Skipping cleaning.")
            detector = None

        try:
            # Determine device for lama based on gpu flag
            device = 'cuda' if args.gpu else 'cpu'
            inpainter = SimpleLama(device=device)
        except Exception as e:
            print(f"Failed to initialize Inpainter: {e}. Skipping cleaning.")
            inpainter = None

        if detector and inpainter:
            listings_final = remove_watermarks_batch(
                listings_with_paths=listings_with_downloads,
                cleaned_dir=args.cleaned_dir,
                detector=detector,
                inpainter=inpainter,
                mask_padding=args.mask_padding
            )
        else:
            print("Skipping cleaning step due to initialization errors.")
            listings_final = listings_with_downloads # Pass through
            # Ensure 'cleaned_paths' exists even if empty when skipping/failed
            for listing in listings_final:
                if 'cleaned_paths' not in listing:
                    listing['cleaned_paths'] = []
    else:
        print("Skipping watermark removal step as requested.")
        listings_final = listings_with_downloads # Pass through if skipping
        # Ensure 'cleaned_paths' exists even if empty when skipping
        for listing in listings_final:
            if 'cleaned_paths' not in listing:
                listing['cleaned_paths'] = []

    # 4. Save Final Data
    print(f"\nSaving final data for {len(listings_final)} listings to {args.output_json}")
    try:
        # Clean up potentially large objects before saving if needed (e.g., mask data if accidentally included)
        for listing in listings_final:
            # Example: remove any temporary keys if they exist
            listing.pop('temp_key', None)

        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(listings_final, f, indent=2, ensure_ascii=False)
        print("Final data saved successfully.")
    except IOError as e:
        print(f"Error saving final data JSON: {e}")
    except TypeError as e:
         print(f"Error serializing final data to JSON (might contain non-serializable objects): {e}")

    print("--- Script Finished --- ")

if __name__ == "__main__":
    main()

