import requests
import json
import os
import dotenv

# Load environment variables
dotenv.load_dotenv()
METABASE_URL = os.getenv("METABASE_URL")

from metabase_client import MetabaseClient

# Initialize Metabase client
metabase = MetabaseClient()

# Ensure export folder exists
os.makedirs("metabase/dashboards", exist_ok=True)

# Get all dashboards
dashboards = requests.get(f"{METABASE_URL}/api/dashboard", headers=metabase.headers).json()

for dashboard in dashboards:
    dashboard_id = dashboard["id"]
    dashboard_name = dashboard["name"].replace(" ", "_") + ".json"

    # Fetch full dashboard details
    dashboard_data = requests.get(f"{METABASE_URL}/api/dashboard/{dashboard_id}", headers=metabase.headers).json()

    # Ensure `description` is stored as `None` if empty
    description = dashboard_data.get("description", None)
    if description == "":
        description = None

    # Save the dashboard with all necessary attributes
    with open(f"metabase/dashboards/{dashboard_name}", "w", encoding="utf-8") as f:
        json.dump(
            {
                "id": dashboard_data["id"],
                "name": dashboard_data["name"],
                "description": description,
                "collection_id": dashboard_data.get("collection_id", None),
                "collection_position": dashboard_data.get("collection_position", None),
                "parameters": dashboard_data.get("parameters", []),
                "cache_ttl": dashboard_data.get("cache_ttl", None),
                "ordered_cards": dashboard_data.get("ordered_cards", []),
            },
            f,
            indent=4,
        )

print("Dashboards exported successfully.")
