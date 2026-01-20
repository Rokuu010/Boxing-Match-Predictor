# src/utils.py

# In this file, I've gathered all the helper and utility functions I use across the project.
# My goal was to keep my main scripts (like train.py and app.py) focused on the core logic,
# so I moved reusable code like logging setup, data cleaning, and web scraping
# into this central location.

import logging
import re
import pandas as pd
from bs4 import BeautifulSoup
import wikipedia
import requests
import shutil
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service


def setup_logging():
    """I created this simple function to set up a clean and consistent logging format."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# Data Cleaning & Parsing Helpers
# I wrote these helper functions to clean messy text from various sources
# and reliably extract the numerical values I need.

def clean_stat(value):
    """A general-purpose function to remove citations and special characters."""
    if pd.isna(value): return None
    value = str(value)
    value = re.sub(r"\[.*?\]", "", value)
    value = value.replace("½", ".5").replace("⁄", "/").replace("+", "")
    return value.strip()

def parse_height(height_str):
    """Extracts height in centimetres from a string."""
    if not height_str: return None
    m = re.search(r'(\d+)\s*cm', height_str)
    return float(m.group(1)) if m else None

def parse_reach(reach_str):
    """Extracts reach in centimetres from a string, handling inches from BoxRec."""
    if not reach_str: return None
    m_in = re.search(r'(\d+)″', reach_str)
    if m_in:
        return float(m_in.group(1)) * 2.54
    m_cm = re.search(r'(\d+)\s*cm', reach_str)
    return float(m_cm.group(1)) if m_cm else None

def parse_weight(weight_str):
    """
    I wrote this to extract a fighter's weight from a string and ensure it's
    always in pounds (lbs). It first looks for kilograms and converts them,
    then looks for pounds. This makes my data consistent.
    """
    if not weight_str: return None
    kg_match = re.search(r'([\d.]+)\s*kg', weight_str)
    if kg_match:
        return float(kg_match.group(1)) * 2.20462
    lbs_match = re.search(r'([\d.]+)\s*lbs', weight_str)
    if lbs_match:
        return float(lbs_match.group(1))
    return None

def parse_age_from_dob(dob_str):
    """I created this to parse a date of birth and calculate the current age."""
    if not dob_str: return None
    match = re.search(r'(\d{4}-\d{2}-\d{2})', dob_str) or re.search(r'(\d{4})', dob_str)
    if match:
        birth_year = int(match.group(1)[:4])
        return datetime.now().year - birth_year
    return None

def parse_wins(wins_str):
    """I wrote this to extract the number of wins from the infobox text."""
    if not wins_str: return None
    match = re.match(r'(\d+)', wins_str)
    return int(match.group(1)) if match else None

def parse_kos(kos_str):
    """
    This now only looks for the specific
    (X KOs) pattern and will not incorrectly use the total win count.
    """
    if not kos_str: return None
    match = re.search(r'\((\d+)\s*KOs?\)', kos_str)
    return int(match.group(1)) if match else None


# Web Scraping Helpers (Now using Selenium for BoxRec)

def _get_selenium_driver():
    """
    I created this helper to set up the Selenium browser driver.
    UPDATED: Now uses the system-installed 'chromium-driver' from packages.txt
    to avoid the 'Exec format error' caused by webdriver-manager.
    """
    chrome_options = Options()
    # 'headless' mode means the browser runs in the background without a visible window.
    # This is MANDATORY for Streamlit Cloud.
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Finds the driver installed via apt-get (packages.txt)
    # This usually looks in /usr/bin/chromedriver or finds it automatically.
    driver_path = shutil.which("chromedriver") or "/usr/bin/chromedriver"

    service = Service(driver_path)

    return webdriver.Chrome(service=service, options=chrome_options)

def fetch_boxrec_stats(name):
    """
    This is my new, more powerful scraper for BoxRec that uses Selenium to avoid
    being blocked. It's a bit slower but much more reliable.
    """
    driver = None
    try:
        driver = _get_selenium_driver()
        search_url = f"https://boxrec.com/en/search?search%5Bquery%5D={name.replace(' ', '+')}"
        driver.get(search_url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        first_result = soup.find('a', href=lambda href: href and '/en/proboxer/' in href)
        if not first_result: return {}

        profile_url = f"https://boxrec.com{first_result['href']}"
        driver.get(profile_url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        stats = {}
        profile_table = soup.find('table', {'class': 'profileTable'})
        if profile_table:
            for row in profile_table.find_all('tr'):
                label_cell = row.find('td', {'class': 'labelCell'})
                if label_cell:
                    label = label_cell.text.strip().lower()
                    value_cell = label_cell.find_next_sibling('td')
                    if value_cell:
                        value = value_cell.text.strip()
                        if 'born' in label: stats['age'] = parse_age_from_dob(value)
                        if 'reach' in label: stats['reach'] = parse_reach(value)
                        if 'height' in label: stats['height'] = parse_height(value)

            wins_cell = soup.find('td', {'class': 'textWon'})
            if wins_cell:
                stats['wins'] = int(wins_cell.text)
                kos_cell = wins_cell.find_next_sibling('td')
                if kos_cell:
                    stats['wins_by_ko'] = int(kos_cell.text)
        return stats
    except Exception as e:
        # Logging warning so i know if scraping fails, but the app won't crash
        logging.warning(f"Selenium scrape for {name} failed: {e}")
        return {}
    finally:
        # Close the browser driver when I'm done.
        if driver:
            driver.quit()

def wiki_search(name):
    """
    This function takes a fighter's name and searches Wikipedia for the most
    likely page title. I appended 'boxer' to the search to get more accurate results.
    """
    try:
        results = wikipedia.search(name + " boxer")
        return results[0] if results else None
    except Exception:
        return None

def fetch_wiki_stats(name):
    """
    This is my fallback scraper. It gets stats from Wikipedia if BoxRec fails.
    """
    page_title = wiki_search(name)
    if not page_title: return {}
    try:
        html_content = wikipedia.page(page_title, auto_suggest=False).html()
        soup = BeautifulSoup(html_content, "html.parser")
        info = {}
        table = soup.find("table", {"class": "infobox"})
        if not table: return {}
        for row in table.find_all("tr"):
            header, data = row.find("th"), row.find("td")
            if not header or not data: continue

            key = header.text.strip().lower()
            val = data.text.strip()

            if "height" in key: info["height"] = parse_height(clean_stat(val))
            if "reach" in key: info["reach"] = parse_reach(clean_stat(val))
            if "weight" in key: info["weight"] = parse_weight(clean_stat(val))
            if "born" in key: info["age"] = parse_age_from_dob(clean_stat(val))
            if key == "wins":
                info["wins"] = parse_wins(clean_stat(val))
                info["wins_by_ko"] = parse_kos(clean_stat(val))
        return info
    except Exception:
        return {}

def get_fighter_data(name):
    """
    My main data fetching function it works by:
    1. Trying to get stats from BoxRec first (primary source).
    2. Trys to get stats from Wikipedia second (fallback source).
    3. Combines the results, giving preference to the more reliable BoxRec data.
    """
    logging.info(f"Fetching live data for {name}...")
    boxrec_stats = fetch_boxrec_stats(name)
    wiki_stats = fetch_wiki_stats(name)

    # This combine the dictionaries while prioritising the BoxRec data.
    final_stats = {
        "name": name,
        "height": boxrec_stats.get("height") or wiki_stats.get("height"),
        "reach": boxrec_stats.get("reach") or wiki_stats.get("reach"),
        "weight": wiki_stats.get("weight"),
        "age": boxrec_stats.get("age") or wiki_stats.get("age"),
        "wins": boxrec_stats.get("wins") or wiki_stats.get("wins"),
        "wins_by_ko": boxrec_stats.get("wins_by_ko") or wiki_stats.get("wins_by_ko"),
    }
    return final_stats