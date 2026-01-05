"""
Scraper for Riksbanken Penningpolitisk rapport (Monetary Policy Reports).

Downloads PDF reports in Swedish from riksbank.se.
"""

import os
import time
from pathlib import Path

import requests
from tqdm import tqdm

# Base URL pattern for Riksbanken reports
# Pattern: /globalassets/media/rapporter/ppr/penningpolitiska-rapporter-och-uppdateringar/svenska/{year}/penningpolitisk-rapport-{month}-{year}.pdf
BASE_URL = "https://www.riksbank.se/globalassets/media/rapporter/ppr/penningpolitiska-rapporter-och-uppdateringar/svenska"

# Riksbanken publishes full reports ~4 times per year (typically Feb/Mar, Jun, Sep, Nov/Dec)
# and updates in between. We want the full "penningpolitisk-rapport" not "uppdatering"
REPORT_MONTHS = ["februari", "mars", "juni", "september", "november", "december"]

# Years to scrape (going back to get 10+ reports)
YEARS = [2025, 2024, 2023, 2022, 2021, 2020]

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "raw"


def build_report_urls() -> list[dict]:
    """
    Build a list of potential report URLs to try.

    Returns list of dicts with 'url', 'year', 'month' keys.
    """
    urls = []

    for year in YEARS:
        for month in REPORT_MONTHS:
            url = f"{BASE_URL}/{year}/penningpolitisk-rapport-{month}-{year}.pdf"
            urls.append({
                "url": url,
                "year": year,
                "month": month,
                "filename": f"penningpolitisk-rapport-{month}-{year}.pdf"
            })

    return urls


def download_pdf(url: str, output_path: Path, timeout: int = 30) -> bool:
    """
    Download a PDF from URL to output_path.

    Returns True if successful, False otherwise.
    """
    try:
        response = requests.get(url, timeout=timeout, stream=True)

        if response.status_code == 200:
            # Verify it's actually a PDF
            content_type = response.headers.get("content-type", "")
            if "pdf" not in content_type.lower() and not url.endswith(".pdf"):
                print(f"  Skipping {url} - not a PDF (content-type: {content_type})")
                return False

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        else:
            return False

    except requests.RequestException as e:
        print(f"  Error downloading {url}: {e}")
        return False


def scrape_riksbanken_reports(max_reports: int = 15) -> list[Path]:
    """
    Download Riksbanken monetary policy reports.

    Args:
        max_reports: Maximum number of reports to download.

    Returns:
        List of paths to downloaded PDFs.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    urls = build_report_urls()
    downloaded = []

    print(f"Attempting to download up to {max_reports} Riksbanken reports...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    for item in tqdm(urls, desc="Checking reports"):
        if len(downloaded) >= max_reports:
            break

        output_path = OUTPUT_DIR / item["filename"]

        # Skip if already downloaded
        if output_path.exists():
            print(f"  Already exists: {item['filename']}")
            downloaded.append(output_path)
            continue

        success = download_pdf(item["url"], output_path)

        if success:
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  Downloaded: {item['filename']} ({size_mb:.1f} MB)")
            downloaded.append(output_path)

        # Be polite to the server
        time.sleep(0.5)

    print()
    print(f"Successfully downloaded {len(downloaded)} reports.")

    # Calculate total size
    total_size = sum(p.stat().st_size for p in downloaded)
    print(f"Total size: {total_size / (1024 * 1024):.1f} MB")

    return downloaded


def main():
    """Main entry point."""
    reports = scrape_riksbanken_reports(max_reports=15)

    if len(reports) < 10:
        print()
        print("WARNING: Downloaded fewer than 10 reports.")
        print("Consider checking if URLs have changed or adding more years.")

    return reports


if __name__ == "__main__":
    main()
