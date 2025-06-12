import requests
import time
import re
import datetime
import os
from urllib.parse import urljoin, urlparse, parse_qs, urlencode
from bs4 import BeautifulSoup

class OpenSooqScraper:
    """
    A parser module for OpenSooq.com HTML content to extract category information,
    parse listing details from search results pages, and handle pagination.
    Does not perform network requests.
    """

    BASE_URL = "https://jo.opensooq.com" # Assuming Jordan, adjust if needed

    def __init__(self, html_content: str, base_url: str = "https://jo.opensooq.com"):
        """
        Initializes the parser with the HTML content of an OpenSooq page.

        Args:
            html_content (str): The full HTML content of the page.
            base_url (str): The base URL for resolving relative links (e.g., https://jo.opensooq.com)
        """
        self.soup = BeautifulSoup(html_content, 'html.parser')
        self.BASE_URL = base_url

    def _clean_text(self, text: str) -> str:
        """Helper to clean text from common Arabic formatting characters."""
        if text:
            # Remove ZWNJ, ZWSP, LTR/RTL marks, and strip whitespace
            cleaned = re.sub(r'[\u200c\u200b\u200e\u200f]', '', text)
            return cleaned.strip()
        return ""

    def _calculate_absolute_datetime(self, posted_date_str: str, fetch_time: datetime.datetime) -> str | None:
        """
        Calculates the approximate absolute datetime of a post based on its
        relative posted date string (Arabic and English) and the page's fetch time.
        """
        if not posted_date_str:
            return None

        posted_datetime = None
        now = fetch_time

        # Clean the input string first
        posted_date_str_cleaned = self._clean_text(posted_date_str).lower()

        try:
            # Arabic Relative Dates
            if "الآن" in posted_date_str_cleaned or "now" in posted_date_str_cleaned:
                posted_datetime = now
            elif "أمس" in posted_date_str_cleaned or "yesterday" in posted_date_str_cleaned:
                posted_datetime = now - datetime.timedelta(days=1)
            elif "دقيقة" in posted_date_str_cleaned or "دقائق" in posted_date_str_cleaned or "minute" in posted_date_str_cleaned:
                minutes_match = re.search(r'(\d+)\s+(?:دقيقة|دقائق|minute|minutes)', posted_date_str_cleaned)
                if minutes_match:
                    minutes_ago = int(minutes_match.group(1))
                    posted_datetime = now - datetime.timedelta(minutes=minutes_ago)
            elif "ساعة" in posted_date_str_cleaned or "ساعات" in posted_date_str_cleaned or "hour" in posted_date_str_cleaned:
                hours_match = re.search(r'(\d+)\s+(?:ساعة|ساعات|hour|hours)', posted_date_str_cleaned)
                if hours_match:
                    hours_ago = int(hours_match.group(1))
                    posted_datetime = now - datetime.timedelta(hours=hours_ago)
            elif "يوم" in posted_date_str_cleaned or "أيام" in posted_date_str_cleaned or "day" in posted_date_str_cleaned:
                days_match = re.search(r'(\d+)\s+(?:يوم|أيام|day|days)', posted_date_str_cleaned)
                if days_match:
                    days_ago = int(days_match.group(1))
                    posted_datetime = now - datetime.timedelta(days=days_ago)
            # Absolute Date format "DD-MM-YYYY"
            elif re.match(r'\d{1,2}-\d{1,2}-\d{4}', posted_date_str_cleaned):
                posted_datetime = datetime.datetime.strptime(posted_date_str_cleaned, '%d-%m-%Y').replace(
                    hour=now.hour, minute=now.minute, second=now.second, microsecond=now.microsecond, tzinfo=datetime.timezone.utc
                )
            # Absolute Date format "Month DD" (e.g., "May 29") - assumes current year
            elif re.match(r'[a-zA-Z]+\s+\d{1,2}', posted_date_str_cleaned):
                 try:
                     # Attempt to parse with current year
                     date_with_year = f"{posted_date_str_cleaned} {now.year}"
                     posted_datetime = datetime.datetime.strptime(date_with_year, '%b %d %Y').replace(
                         hour=now.hour, minute=now.minute, second=now.second, microsecond=now.microsecond, tzinfo=datetime.timezone.utc
                     )
                     # If parsed date is in the future, assume it was last year
                     if posted_datetime > now:
                         posted_datetime = posted_datetime.replace(year=now.year - 1)
                 except ValueError:
                     pass # Ignore if format doesn't match

        except Exception as e:
            print(f"Error parsing date string '{posted_date_str}': {e}")
            return None # Return None if parsing fails

        return posted_datetime.isoformat() if posted_datetime else None

    def _get_high_res_image_url(self, preview_url: str) -> str | None:
        """
        Converts a preview image URL to the highest quality version available.
        Handles multiple OpenSooq image URL patterns and ensures maximum resolution.
        """
        if not preview_url or not isinstance(preview_url, str):
            return None

        # Make a copy to work with
        high_res_url = preview_url

        # Dictionary of resolution patterns to replace with highest quality
        resolution_patterns = {
            "previews/640x480/": "previews/2048x0/",
            "previews/0x720/": "previews/2048x0/",
            "previews/0x240/": "previews/2048x0/",
            "previews/320x240/": "previews/2048x0/",
            "previews/480x360/": "previews/2048x0/",
            "previews/800x600/": "previews/2048x0/",
            # Add more patterns as needed
        }

        # Apply resolution upgrades
        for low_res, high_res in resolution_patterns.items():
            if low_res in high_res_url:
                high_res_url = high_res_url.replace(low_res, high_res)
                break

        # Handle thumbnail patterns (common in OpenSooq)
        if "/thumbs/" in high_res_url:
            high_res_url = high_res_url.replace("/thumbs/", "/previews/0x1080/")

        # Handle small image patterns
        if "/small/" in high_res_url:
            high_res_url = high_res_url.replace("/small/", "/previews/0x1080/")

        # Remove webp extension and replace with jpg for better compatibility
        # if high_res_url.endswith(".webp"):
        #     high_res_url = high_res_url[:-5] + ".jpg"

        # Ensure we have a proper image extension
        parsed_url = urlparse(high_res_url)
        path = parsed_url.path

        # Check if path has a proper image extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp','.webp']
        has_valid_extension = any(path.lower().endswith(ext) for ext in valid_extensions)

        # if not has_valid_extension:
        #     # Remove any existing extension and add .jpg
        #     base_path = os.path.splitext(path)[0]
        #     high_res_url = high_res_url.replace(path, base_path + '.jpg')

        # Final quality check - ensure we're using the highest resolution pattern
        # if "previews/" in high_res_url and "2048x0" not in high_res_url:
        #     # Extract the filename and rebuild with highest quality
        #     filename = os.path.basename(urlparse(high_res_url).path)
        #     base_url_parts = high_res_url.split('/previews/')[0]
        #     high_res_url = f"{base_url_parts}/previews/2048x0/{filename}"

        return high_res_url

    def parse_listings(self, fetch_time: datetime.datetime) -> list:
        """
        Parses listing summaries from a search results page.

        Args:
            fetch_time (datetime.datetime): The time the page HTML was fetched.

        Returns:
            list: A list of dictionaries, each representing a listing summary.
        """
        listings = []
        listing_elements = self.soup.find_all('div', class_=lambda c: c and 'postLi' in c and 'mb-16' in c)
        if not listing_elements:
             listing_elements = self.soup.find_all('a', class_='postListItemData') # Fallback

        print(f"Found {len(listing_elements)} potential listing elements.")

        for item in listing_elements:
            listing = {}
            link_tag = item.find('a', class_='postListItemData') if item.name == 'div' else item
            if not link_tag:
                continue

            listing['url'] = urljoin(self.BASE_URL, link_tag.get('href', ''))
            if not listing['url'] or not listing['url'].startswith('http'):
                continue

            title_tag = link_tag.find('h2', class_='trimTwoLines')
            listing['title'] = self._clean_text(title_tag.text) if title_tag else None

            img_tag = link_tag.find('img', class_='radius-8')
            preview_image_url = None
            if img_tag:
                preview_image_url = img_tag.get('data-src') or img_tag.get('src')
                if preview_image_url and not preview_image_url.startswith('http'):
                    preview_image_url = urljoin(self.BASE_URL, preview_image_url)

            listing['preview_image_url'] = preview_image_url
            listing['high_res_image_url'] = self._get_high_res_image_url(preview_image_url)

            date_span = link_tag.find('span', class_='postDate')
            listing['posted_date_text'] = self._clean_text(date_span.text) if date_span else None
            listing['posted_datetime'] = self._calculate_absolute_datetime(listing['posted_date_text'], fetch_time)

            price_div = link_tag.find('div', class_='priceColor')
            listing['price_text'] = None
            listing['price_value'] = None
            if price_div:
                price_text_raw = self._clean_text(price_div.text)
                listing['price_text'] = price_text_raw
                price_match = re.search(r'[\d,.]+', price_text_raw)
                if price_match:
                    try:
                        price_str = price_match.group(0).replace(',', '')
                        listing['price_value'] = float(price_str)
                    except ValueError:
                        pass

            location_div = link_tag.find('div', class_='darkGrayColor')
            listing['city'] = None
            listing['neighborhood'] = None
            if location_div:
                location_parts = [self._clean_text(part) for part in location_div.get_text(separator='<!-- -->').split('<!-- -->') if self._clean_text(part)]
                listing['city'] = location_parts[0] if location_parts else None
                listing['neighborhood'] = location_parts[1] if len(location_parts) > 1 else None

            details_p = link_tag.find('p', class_=lambda c: c and 'flexSpaceBetween' in c and 'gap-10' in c)
            listing['details'] = []
            if details_p:
                details_text = self._clean_text(details_p.text)
                listing['details'] = [self._clean_text(d) for d in re.split(r'\s*,\s*', details_text) if self._clean_text(d)]

            listing['is_vip'] = bool(link_tag.find('svg', {'data-name': 'iconVip'}))
            listing['is_turbo'] = bool(link_tag.find('svg', {'data-name': 'iconTurbo'}))
            listing['has_video'] = bool(link_tag.find('svg', {'data-name': 'IconVideo'}))
            listing['photo_count'] = None
            media_count_span = link_tag.find('span', class_='postListItemMedia', string=re.compile(r'\d+'))
            if media_count_span:
                count_match = re.search(r'(\d+)', media_count_span.text)
                if count_match:
                    try:
                        listing['photo_count'] = int(count_match.group(1))
                    except ValueError:
                        pass

            # Add listing only if essential info is present
            if listing.get('url') and listing.get('title') and listing.get('high_res_image_url'):
                listings.append(listing)

        print(f"Successfully parsed {len(listings)} listings.")
        return listings

    def get_pagination_info(self) -> dict:
        """
        Extracts pagination information (next page URL, last page number).
        """
        pagination_info = {
            'next_page_url': None,
            'last_page_number': None
        }
        pagination_div = self.soup.find('div', id='pagination')
        if pagination_div:
            next_page_arrow = pagination_div.find('a', {'data-id': 'nextPageArrow'})
            if next_page_arrow and 'disabled' not in next_page_arrow.get('class', []) and next_page_arrow.get('href'):
                pagination_info['next_page_url'] = urljoin(self.BASE_URL, next_page_arrow['href'])

            page_links = pagination_div.find_all('a', {'data-id': re.compile(r'page_\d+')})
            max_page = 0
            for link in page_links:
                try:
                    page_num = int(link.text.strip())
                    if page_num > max_page:
                        max_page = page_num
                except (ValueError, AttributeError):
                    continue
            if max_page > 0:
                pagination_info['last_page_number'] = max_page
            else:
                last_page_arrow = pagination_div.find('a', {'data-id': 'lastPageArrow'})
                if last_page_arrow and last_page_arrow.get('href'):
                    parsed_url = urlparse(last_page_arrow['href'])
                    query_params = parse_qs(parsed_url.query)
                    if 'page' in query_params and query_params['page'][0].isdigit():
                        pagination_info['last_page_number'] = int(query_params['page'][0])
        return pagination_info

    @staticmethod
    def construct_search_url(base_category_url: str, search_term: str = None, page: int = 1, sort_recent: bool = False) -> str:
        """
        Constructs a search URL for a specific page, optional search term, and sorting.

        Args:
            base_category_url (str): The starting URL for the category.
            search_term (str, optional): A search term to add. Defaults to None.
            page (int, optional): The page number. Defaults to 1.
            sort_recent (bool, optional): Whether to sort by most recent. Defaults to False.

        Returns:
            str: The constructed URL.
        """
        parsed_url = urlparse(base_category_url)
        query_params = parse_qs(parsed_url.query)

        if search_term:
            query_params['search'] = ['true']
            query_params['term'] = [search_term]
        else:
            query_params.pop('search', None)
            query_params.pop('term', None)

        if page > 1:
            query_params['page'] = [str(page)]
        elif 'page' in query_params:
            del query_params['page']

        if sort_recent:
            query_params['search'] = ['true'] # 'search=true' seems required for sorting
            query_params['sort_code'] = ['recent']
        elif 'sort_code' in query_params:
            del query_params['sort_code']

        new_query = urlencode(query_params, doseq=True)
        return parsed_url._replace(query=new_query).geturl()

    def parse_listing_details(self, fetch_time: datetime.datetime) -> dict:
        """
        Parses detailed information from a single listing page.

        Args:
            fetch_time (datetime.datetime): The exact time the page was fetched.

        Returns:
            dict: A dictionary containing all extracted details from the listing page.
        """
        details = {}

        # 1. Title
        title_tag = self.soup.find('h1', class_='postViewTitle')
        details['title'] = self._clean_text(title_tag.text) if title_tag else None

        # 2. Price
        price_div = self.soup.find('div', class_='priceColor')
        if price_div:
            price_text = self._clean_text(price_div.text)
            details['price'] = re.sub(r'[^\d,.]', '', price_text).replace(',', '')
            try:
                details['price_value'] = float(details['price'])
            except ValueError:
                details['price_value'] = None
        else:
            details['price'] = None
            details['price_value'] = None

        # 3. Posted Date
        # posted_date_span = self.soup.find('div', class_='flex gap-5 alignItems')
        # if posted_date_span:
        #     posted_date_span = posted_date_span.find('span', class_='darkGrayColor')
        # details['posted_date_text'] = self._clean_text(posted_date_span.text) if posted_date_span else None
        # if details['posted_date_text']:
        #     details['posted_datetime'] = self._calculate_absolute_datetime(details['posted_date_text'], fetch_time)
        # else:
        #     details['posted_datetime'] = None

        # 4. Information Section (Key-Value Pairs) - FIXED
        info_section = self.soup.find('section', id='PostViewInformation')
        details['specifications'] = {}
        if info_section:
            for li in info_section.find_all('li'):
                # Handle different types of list items
                if 'fullRow' in li.get('class', []):
                    # Full row items like "المزايا الرئيسية" and "المزايا الإضافية"
                    p_tags = li.find_all('p')
                    if len(p_tags) >= 2:
                        label = self._clean_text(p_tags[0].text)
                        value = self._clean_text(p_tags[1].text)
                        if label and value:
                            details['specifications'][label] = value
                else:
                    # Single info field items
                    label_p = li.find('p', class_='noWrap')
                    label = self._clean_text(label_p.text) if label_p else None
                    
                    # Look for value in different possible containers
                    value_tag = li.find('a', class_='bold') or li.find('span', class_='bold') or li.find('a') or li.find('span')
                    value = self._clean_text(value_tag.text) if value_tag else None
                    
                    if label and value:
                        details['specifications'][label] = value

        # 5. Description
        description_div = self.soup.find('div', {"data-id":"description"})
        if description_div:
            # Extract all text, then clean up extra spaces/newlines
            description_text = description_div.get_text(separator='\n').strip()
            details['description'] = self._clean_text(description_text)
        else:
            details['description'] = None

        # 6. Owner Information
        owner_card = self.soup.find('section', id='PostViewOwnerCard')
        details['owner_info'] = {}
        if owner_card:
            name_tag = owner_card.find('h3', class_='blueColor')
            details['owner_info']['name'] = self._clean_text(name_tag.text) if name_tag else None

            member_since_div = owner_card.find('div', string=re.compile(r'عضو منذ'))
            if member_since_div:
                match = re.search(r'عضو منذ\s*(\d{2}-\d{2}-\d{4})', self._clean_text(member_since_div.text))
                details['owner_info']['member_since'] = match.group(1) if match else None

            response_time_div = owner_card.find('div', string=re.compile(r'وقت التجاوب'))
            if response_time_div:
                details['owner_info']['response_time'] = self._clean_text(response_time_div.text).replace('وقت التجاوب', '').strip()
            else:
                details['owner_info']['response_time'] = None

            # Owner Rating (number of yellow stars)
            rating_div = owner_card.find('div', class_='fiveStarsRating')
            if rating_div:
                yellow_stars = rating_div.find('div', class_='yellowStars')
                if yellow_stars:
                    details['owner_info']['rating_stars'] = len(yellow_stars.find_all('svg', {'fill': '#fbce0f'}))
                else:
                    details['owner_info']['rating_stars'] = 0
            else:
                details['owner_info']['rating_stars'] = 0

        # 7. Location and Map Coordinates
        location_section = self.soup.find('section', id='postViewLocation')
        if location_section:
            location_link = location_section.find('a', class_='blackColor', data_id='location')
            details['location_text'] = self._clean_text(location_link.text) if location_link else None

            map_link = location_section.find('a', {'data-ghost': 'map_google'})
            if map_link and map_link.get('href'):
                map_url_parsed = urlparse(map_link['href'])
                map_query_params = parse_qs(map_url_parsed.query)
                if 'query' in map_query_params and map_query_params['query']:
                    coords = map_query_params['query'][0].split(',')
                    if len(coords) == 2:
                        try:
                            details['latitude'] = float(coords[0])
                            details['longitude'] = float(coords[1])
                        except ValueError:
                            pass

        # 8. Phone Numbers
        details['phone_numbers'] = []
        # From call button
        call_btn = self.soup.find('button', data_id='call_btn')
        if call_btn:
            phone_span = call_btn.find('span', class_='ltr')
            if phone_span and phone_span.text.strip():
                details['phone_numbers'].append(self._clean_text(phone_span.text))

        # From description (using regex to find common Jordanian numbers)
        if details['description']:
            # Jordanian mobile numbers typically start with 07 and are 10 digits.
            # You might need to adjust regex based on observed number formats.
            phone_regex = r'(07[789]\d{7}|06\d{7})' # Matches 07X-XXXXXXX or 06-XXXXXXX
            found_numbers = re.findall(phone_regex, details['description'])
            for num in found_numbers:
                cleaned_num = re.sub(r'\D', '', num) # Remove non-digits
                if cleaned_num not in details['phone_numbers']:
                    details['phone_numbers'].append(cleaned_num)

        # 9. Image URLs - Enhanced to ensure all images are high quality
        details['image_urls'] = []

        # Find all thumbnail images as they represent all available images
        thumbnail_images = self.soup.select('#postViewGallery .image-gallery-thumbnail-image')
        for img_tag in thumbnail_images:
            if img_tag.get('src'):
                original_url = urljoin(self.BASE_URL, img_tag['src'])
                high_res_url = self._get_high_res_image_url(original_url)
                if high_res_url:
                    details['image_urls'].append(high_res_url)

        # Fallback to main gallery images if no thumbnails found
        if not details['image_urls']:
            main_gallery_images = self.soup.select('#postViewGallery .image-gallery-slides .image-gallery-image')
            for img_tag in main_gallery_images:
                if img_tag.get('src'):
                    original_url = urljoin(self.BASE_URL, img_tag['src'])
                    high_res_url = self._get_high_res_image_url(original_url)
                    if high_res_url:
                        details['image_urls'].append(high_res_url)
                        print(f"Converted to high-res: {high_res_url}")

        # Additional fallback - check for any other image patterns
        if not details['image_urls']:
            all_images = self.soup.find_all('img')
            for img_tag in all_images:
                src = img_tag.get('src') or img_tag.get('data-src')
                if src and ('preview' in src or 'image' in src):
                    original_url = urljoin(self.BASE_URL, src)
                    high_res_url = self._get_high_res_image_url(original_url)
                    if high_res_url and high_res_url not in details['image_urls']:
                        details['image_urls'].append(high_res_url)

        return details
