import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin

# --------- CONFIG ---------
URL = "https://reliefweb.int/updates?advanced-search=(country%3A85)"  # Example: ReliefWeb Pakistan updates
URL = "https://pdma.gos.pk/Documents/Monsoon/Monsoon_2025/Alerts_&_Advisories/Flood Alert Dated 26-09-2025.pdf"
URL = "https://www.ndma.gov.pk/sitreps"
SAVE_DIR = "downloads"  # Folder to save the file

# --------- SCRAPER ---------
def download_latest_pdf(url, save_dir):
    # Step 1: Fetch webpage
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve page: {response.status_code}")
        return
    
    # Step 2: Parse PDF links
    soup = BeautifulSoup(response.text, "html.parser")
    pdf_links = []

    for a in soup.find_all("a", href=True):
        href = a['href']
        if href.lower().endswith(".pdf"):
            full_url = urljoin(url, href)
            pdf_links.append(full_url)
    
    if not pdf_links:
        print("No PDF links found on page.")
        return

    # Step 3: Pick the latest (usually first on the page)
    latest_pdf = pdf_links[0]
    print(f"Found latest PDF: {latest_pdf}")

    # Step 4: Download the file
    pdf_response = requests.get(latest_pdf)
    if pdf_response.status_code == 200:
        os.makedirs(save_dir, exist_ok=True)
        filename = latest_pdf.split("/")[-1]
        file_path = os.path.join(save_dir, filename)
        with open(file_path, "wb") as f:
            f.write(pdf_response.content)
        print(f"✅ Downloaded: {file_path}")
    else:
        print(f"Failed to download PDF: {pdf_response.status_code}")

# Run function

