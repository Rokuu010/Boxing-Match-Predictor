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
from datetime import datetime

def setup_logging():
    """I created this simple function to set up a clean and consistent logging format."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# --- Data Cleaning & Parsing Helpers ---
# I wrote these helper functions to clean messy text from Wikipedia
# and reliably extract the numerical values I need.

def clean_stat(value):
    """A general-purpose function to remove citations and special characters."""
    if pd.isna(value):
        return None
    value = str(value)
    value = re.sub(r"\[.*?\]", "", value)  # Remove bracketed citations like [1]
    value = value.replace("½", ".5").replace("⁄", "/").replace("+", "")
    return value.strip()

def parse_height(height_str):
    """Extracts height in centimetres from a string."""
    if not height_str:
        return None
    m = re.search(r"(\d+)\s*cm", height_str)
    return float(m.group(1)) if m else None

def parse_reach(reach_str):
    """Extracts reach in centimetres from a string."""
    if not reach_str:
        return None
    m = re.search(r"(\d+)\s*cm", reach_str)
    return float(m.group(1)) if m else None

def parse_weight(weight_str):
    """
    I wrote this to extract a fighter's weight from a string and ensure it's
    always in pounds (lbs). It first looks for kilograms and converts them,
    then looks for pounds. This makes my data consistent.
    """
    if not weight_str:
        return None
    # First, I look for a value in kilograms (kg) and convert it to pounds.
    kg_match = re.search(r"([\d.]+)\s*kg", weight_str)
    if kg_match:
        kilograms = float(kg_match.group(1))
        return kilograms * 2.20462
    # If I don't find kg, I look for a value in pounds (lbs).
    lbs_match = re.search(r"([\d.]+)\s*lbs", weight_str)
    if lbs_match:
        return float(lbs_match.group(1))
    return None

def parse_age_from_dob(dob_str):
    """I created this to parse a date of birth and calculate the current age."""
    if not dob_str:
        return None
    # I use a regular expression to find the birth date, typically in (YYYY-MM-DD) format.
    match = re.search(r'\((\d{4})-\d{2}-\d{2}\)', dob_str)
    if match:
        birth_year = int(match.group(1))
        current_year = datetime.now().year
        return current_year - birth_year
    return None

def parse_wins(wins_str):
    """I wrote this to extract the number of wins from the infobox text."""
    if not wins_str:
        return None
    # The number of wins is usually the first integer in the string.
    match = re.match(r'(\d+)', wins_str)
    if match:
        return int(match.group(1))
    return None


# --- Wikipedia Scraping Helpers ---
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
    This is my main scraping function. It now fetches height, reach, weight,
    age, and total wins from a fighter's Wikipedia infobox.
    """
    page_title = wiki_search(name)
    if not page_title:
        return {}
    try:
        html_content = wikipedia.page(page_title, auto_suggest=False).html()
        soup = BeautifulSoup(html_content, "html.parser")
        info = {}
        table = soup.find("table", {"class": "infobox"})
        if not table:
            return {}
        for row in table.find_all("tr"):
            header = row.find("th")
            data = row.find("td")
            if not header or not data:
                continue

            key = header.text.strip().lower()
            val = data.text.strip()

            if "height" in key: info["height"] = parse_height(clean_stat(val))
            if "reach" in key: info["reach"] = parse_reach(clean_stat(val))
            if "weight" in key: info["weight"] = parse_weight(clean_stat(val))
            # I added logic to find the 'Born' and 'Wins' rows.
            if "born" in key: info["age"] = parse_age_from_dob(clean_stat(val))
            if key == "wins": info["wins"] = parse_wins(clean_stat(val))
        return info
    except Exception:
        return {}

def get_fighter_data(name):
    """A simple wrapper to fetch and return all available stats for a single fighter."""
    stats = fetch_wiki_stats(name)
    return {
        "name": name,
        "height": stats.get("height"),
        "reach": stats.get("reach"),
        "weight": stats.get("weight"),
        "age": stats.get("age"),
        "wins": stats.get("wins"),
    }