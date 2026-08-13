import os
import hashlib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from rag_engine import index_single_pdf, DOC_FOLDER

# ----------------- CONFIGURATION -----------------
# Expanded seed sections for SNJB Engineering (Homepage + Department/Exam Portals)
TARGET_SECTIONS = [
    os.getenv("COLLEGE_HOMEPAGE_URL", "https://www.snjb.org/engineering/"),
    "https://www.snjb.org/engineering/Academics/syllabus",
    "https://www.snjb.org/engineering/Examination/notices",
    "https://www.snjb.org/engineering/Downloads/",
    "https://www.snjb.org/engineering/Computer_engineering/comp_syllabus",
    "https://www.snjb.org/engineering/Aids_engineering/aids_syllabus",
    "https://www.snjb.org/engineering/Mechanical_engineering/mech_syllabus",
    "https://www.snjb.org/engineering/Civil_engineering/civil_syllabus",
    "https://www.snjb.org/engineering/Entc_engineering/entc_syllabus",
]

# Keywords to auto-discover academic subpages from navigation menus
NAV_KEYWORDS = [
    "syllabus", "syllabi", "exam", "examination", "timetable", "time-table",
    "notice", "circular", "academic", "curriculum", "scheme", "announcement",
    "download", "student-corner", "portal", "regulation", "department",
    "computer", "aids", "mechanical", "civil", "entc", "fe", "se", "te", "be",
    "result", "schedule", "timetable"
]

# Academic whitelist required for indexing
ACADEMIC_WHITELIST = [
    "syllabus", "syllabi", "curriculum", "timetable", "time-table",
    "exam", "examination", "regulation", "circular", "guideline", "rule",
    "scheme", "sppu", "fe", "se", "te", "be", "computer", "aids", "mechanical",
    "civil", "entc", "brochure", "academic", "calendar"
]

# Blacklist to filter out non-academic noise
NOISE_BLACKLIST = [
    "sports", "holiday", "canteen", "tender", "quotation", "event",
    "reunion", "cultural", "photo", "fest", "gallery", "sports-day", "alumni"
]

# Headers imitating a full desktop browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}


# ----------------- HTTP SESSION WITH RETRIES -----------------
def get_robust_session():
    """Creates a resilient requests Session with automatic backoff retries."""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=2,  # Retries after 2s, 4s, 8s
        status_forcelist=[500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


HTTP_SESSION = get_robust_session()


# ----------------- HELPER FUNCTIONS -----------------
def compute_sha256(content_bytes):
    """Generates a SHA-256 hash to detect modified or new PDF files."""
    return hashlib.sha256(content_bytes).hexdigest()


def get_base_domain(url):
    """Extracts base domain to ensure crawler stays on the official college domain."""
    return urlparse(url).netloc.lower()


def is_relevant_academic_pdf(link_text, url):
    """Filters out non-academic public notices before downloading."""
    combined_info = f"{link_text} {url}".lower()

    # Reject if blacklisted keyword found
    if any(black in combined_info for black in NOISE_BLACKLIST):
        return False

    # Accept if whitelisted keyword found
    if any(white in combined_info for white in ACADEMIC_WHITELIST):
        return True

    # Default to True for PDFs found on explicit academic subpages
    return True


def is_pdf_link(href, link_text=""):
    """Detects standard .pdf extensions as well as PDF download handlers (.php, .aspx, etc.)."""
    href_lower = href.lower()
    if href_lower.endswith(".pdf"):
        return True
    if "pdf" in href_lower or "download" in href_lower or "file" in href_lower:
        if any(w in link_text.lower() for w in ACADEMIC_WHITELIST):
            return True
    return False


# ----------------- SMART DISCOVERY LOGIC -----------------
def discover_relevant_subpages(seed_urls):
    """Scans seed pages and navigation menus for academic sub-sections."""
    print("🕸️ [Auto-Crawler] Scanning seed pages & departments for sub-sections...")
    discovered_pages = set(seed_urls)

    for seed in seed_urls:
        base_domain = get_base_domain(seed)
        try:
            response = HTTP_SESSION.get(seed, headers=HEADERS, timeout=(15, 30), verify=True)
            if response.status_code != 200:
                print(f"⚠️ [Auto-Crawler] Seed page HTTP {response.status_code}: {seed}")
                continue

            soup = BeautifulSoup(response.text, "html.parser")

            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"].strip()
                link_text = a_tag.get_text().strip().lower()
                full_url = urljoin(seed, href)

                # Stay within college domain and skip direct image/zip assets
                if get_base_domain(full_url) == base_domain and not href.lower().endswith(
                        ('.jpg', '.png', '.zip', '.jpeg')):
                    combined_info = f"{link_text} {href.lower()}"

                    if any(keyword in combined_info for keyword in NAV_KEYWORDS):
                        discovered_pages.add(full_url)

        except Exception as e:
            print(f"⚠️ [Auto-Crawler Warning] Error scanning seed {seed}: {e}")

    print(f"✅ [Auto-Crawler] Total target pages to inspect: {len(discovered_pages)}")
    return discovered_pages


def extract_pdfs_from_page(page_url):
    """Scrapes a specific page and extracts valid academic PDF links."""
    pdf_urls = set()
    try:
        response = HTTP_SESSION.get(page_url, headers=HEADERS, timeout=(15, 30))
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"].strip()
                link_text = a_tag.get_text().strip()

                if is_pdf_link(href, link_text):
                    full_pdf_url = urljoin(page_url, href)
                    if is_relevant_academic_pdf(link_text, full_pdf_url):
                        pdf_urls.add(full_pdf_url)

    except Exception as e:
        print(f"⚠️ [Auto-Crawler Warning] Failed scanning PDFs on {page_url}: {e}")

    return pdf_urls


# ----------------- MAIN PIPELINE -----------------
def check_and_sync_college_docs():
    """Main Crawler Pipeline: Discover, Download, Hash-Check, and Index into Qdrant."""
    print("\n🚀 [Auto-Crawler] Starting College Portal Auto-Discovery Sync...")
    os.makedirs(DOC_FOLDER, exist_ok=True)

    # 1. Discover all relevant subpages from configured section URLs
    target_pages = discover_relevant_subpages(TARGET_SECTIONS)

    # 2. Collect all matching academic PDF URLs across target pages
    all_pdf_urls = set()
    for page in target_pages:
        found_pdfs = extract_pdfs_from_page(page)
        all_pdf_urls.update(found_pdfs)

    print(f"📄 [Auto-Crawler] Total academic PDF files found: {len(all_pdf_urls)}")

    # 3. Download, hash-check, and auto-index into Qdrant Cloud
    indexed_count = 0
    for pdf_url in all_pdf_urls:
        filename = pdf_url.split("/")[-1].split("?")[0]
        if not filename.endswith(".pdf"):
            filename += ".pdf"

        local_path = os.path.join(DOC_FOLDER, filename)
        hash_file = local_path + ".sha256"

        try:
            res = HTTP_SESSION.get(pdf_url, headers=HEADERS, timeout=(15, 45))
            if res.status_code == 200:
                # Confirm response is actually a PDF file
                content_type = res.headers.get("Content-Type", "").lower()
                if "application/pdf" not in content_type and not pdf_url.lower().endswith(".pdf"):
                    continue

                new_hash = compute_sha256(res.content)

                # Skip download & re-indexing if file hash matches existing local record
                if os.path.exists(local_path) and os.path.exists(hash_file):
                    with open(hash_file, "r") as f:
                        if f.read().strip() == new_hash:
                            continue

                # Write file and store new SHA-256 hash
                with open(local_path, "wb") as f:
                    f.write(res.content)
                with open(hash_file, "w") as f:
                    f.write(new_hash)

                print(f"📥 [Auto-Crawler] Downloaded new/updated document: {filename}")

                # Index document chunks into Qdrant Cloud
                chunk_count = index_single_pdf(local_path, filename)
                print(f"✅ [Auto-Crawler] Successfully indexed '{filename}' ({chunk_count} chunks).")
                indexed_count += 1

        except Exception as e:
            print(f"❌ [Auto-Crawler Error] Failed processing {pdf_url}: {e}")

    print(f"🎉 [Auto-Crawler] Sync complete. {indexed_count} new/updated document(s) indexed.\n")


if __name__ == "__main__":
    check_and_sync_college_docs()