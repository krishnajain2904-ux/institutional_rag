import os
import urllib.parse
import requests
from bs4 import BeautifulSoup
from rag_engine import DOC_FOLDER, index_single_file

# Sanitize input URL to strip accidental Markdown links
RAW_URL = os.getenv("COLLEGE_HOMEPAGE_URL", "https://www.snjb.org/engineering/")


def clean_url(url: str) -> str:
    """Removes any bracket formatting like [https://...](https://...) from env vars."""
    url = url.strip()
    if "[" in url and "]" in url:
        url = url.split("]")[0].replace("[", "")
    return url.strip("()'\" ")


COLLEGE_HOMEPAGE_URL = clean_url(RAW_URL)


def check_and_sync_college_docs():
    """Scans college portal for public PDFs and downloads new ones without crashing."""
    target_url = COLLEGE_HOMEPAGE_URL
    if not target_url.startswith("http"):
        print(f"⚠️ Auto-Crawler skipped: Invalid URL '{target_url}'")
        return

    try:
        print(f"🚀 [Auto-Crawler] Starting portal sync at: {target_url}")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(target_url, headers=headers, timeout=15)

        if resp.status_code != 200:
            print(f"⚠️ Portal returned status code {resp.status_code}")
            return

        soup = BeautifulSoup(resp.text, "html.parser")
        pdf_links = []

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if href.lower().endswith(".pdf"):
                full_url = urllib.parse.urljoin(target_url, href)
                pdf_links.append(full_url)

        print(f"📄 [Auto-Crawler] Discovered {len(pdf_links)} PDF documents.")

        # Sync top 3 documents to avoid RAM spikes on free tier
        for pdf_url in pdf_links[:3]:
            filename = os.path.basename(urllib.parse.urlparse(pdf_url).path)
            if not filename:
                continue

            local_path = os.path.join(DOC_FOLDER, filename)
            if not os.path.exists(local_path):
                print(f"📥 [Auto-Crawler] Downloading '{filename}'...")
                pdf_resp = requests.get(pdf_url, headers=headers, timeout=20)
                if pdf_resp.status_code == 200:
                    with open(local_path, "wb") as f:
                        f.write(pdf_resp.content)
                    index_single_file(local_path, filename)

    except Exception as e:
        print(f"⚠️ [Auto-Crawler Warning] Sync encountered issue: {e}")