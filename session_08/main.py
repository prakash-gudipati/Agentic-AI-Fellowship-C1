"""
main.py — Weather Dashboard CLI
Session 08: APIs, HTTP Requests + Environment Variables

This is the entry point. It does NOT contain any API logic or file logic.
Its only job is to: get user input, call the right classes, display the result.

S7 pattern — project structure:
  main.py    ← you are here. Orchestrates everything.
  weather_client.py ← knows how to call the weather API (WeatherClient class)
  result_saver.py   ← knows how to save results to JSON (ResultSaver class)

This separation means:
  - If the API changes, you only edit weather_client.py
  - If the output format changes, you only edit result_saver.py
  - main.py stays the same

Run: python main.py
"""

import os
from dotenv import load_dotenv

from weather_client import WeatherClient, WEATHER_CODES
from result_saver import ResultSaver


# ── Load .env file ─────────────────────────────────────────────────────────
# load_dotenv() reads the .env file and loads every KEY=VALUE pair into
# os.environ so we can read them with os.environ.get().
# If there is no .env file, it does nothing — no crash.
load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────
OUTPUT_FILE = "weather_results.json"


def display_report(location, weather):
    """
    Print a formatted weather report to the terminal.
    This is a plain function (not a method) because it belongs to main.py,
    not to the WeatherClient class. It has no configuration to store on self.
    """
    weather_code = weather.get("weathercode", -1)
    condition    = WEATHER_CODES.get(weather_code, f"Code {weather_code}")

    print("\n" + "=" * 50)
    print(f"  WEATHER REPORT")
    print(f"  {location['city'].upper()}, {location['country'].upper()}")
    print("=" * 50)
    print(f"  Condition   : {condition}")
    print(f"  Temperature : {weather.get('temperature_2m', 'N/A')} °C")
    print(f"  Humidity    : {weather.get('relative_humidity_2m', 'N/A')} %")
    print(f"  Wind Speed  : {weather.get('windspeed_10m', 'N/A')} km/h")
    print("=" * 50 + "\n")


def main():
    print("=" * 50)
    print("  WEATHER DASHBOARD CLI")
    print("  Powered by Open-Meteo — free & open source")
    print("=" * 50)

    # ── ENV VAR DEMO ───────────────────────────────────────────────────────
    # In a real paid API, we would read the key like this.
    # The key is stored in .env — it never appears in this file.
    # os.environ.get() returns the value or the default if the key is not set.
    demo_key = os.environ.get("DEMO_API_KEY", "not set — add it to your .env file")
    print(f"\n[Env demo] DEMO_API_KEY = {demo_key}")
    print("[Env demo] The key is read from .env — never hardcoded here.\n")

    # ── CREATE OUR CLASS INSTANCES ─────────────────────────────────────────
    # S5 pattern: we instantiate one WeatherClient and one ResultSaver.
    # WeatherClient handles all API communication.
    # ResultSaver handles all file I/O.
    # main.py just coordinates them.
    client = WeatherClient(timeout=10)
    saver  = ResultSaver(OUTPUT_FILE)

    # ── GET USER INPUT ─────────────────────────────────────────────────────
    city = input("Enter a city name: ")

    # ── CALL THE API ───────────────────────────────────────────────────────
    # client.fetch() is one method call that runs the full pipeline:
    # city name → geocode → latitude/longitude → current weather → two dicts
    location, weather = client.fetch(city)

    if not location:
        print("Could not look up that city. Please check the spelling and try again.")
        return

    if not weather:
        print("Location found but weather data is unavailable. Please try again.")
        return

    # ── DISPLAY + SAVE ─────────────────────────────────────────────────────
    display_report(location, weather)

    saved = saver.save(location, weather, WEATHER_CODES)
    print(f"Result saved successfully: {saved['city']}, {saved['country']} at {saved['timestamp'][:19]}")
    print(f"Open {OUTPUT_FILE} to see all saved results.\n")


if __name__ == "__main__":
    main()
