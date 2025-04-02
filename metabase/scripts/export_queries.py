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

# Ensure export folders exist
os.makedirs("metabase/queries", exist_ok=True)
os.makedirs("metabase/models", exist_ok=True)

# Get all saved queries and models
questions = requests.get(f"{METABASE_URL}/api/card", headers=metabase.headers).json()

for question in questions:
    if "dataset_query" in question and "type" in question["dataset_query"]:
        query_id = question["id"]
        query_name = question["name"].replace(" ", "_") + ".json"

        # Fetch full query details
        query_data = requests.get(f"{METABASE_URL}/api/card/{query_id}", headers=metabase.headers).json()

        # Determine if it's a model or a query
        is_model = query_data.get("dataset", False)  # Models have `"dataset": True`
        export_folder = "metabase/models" if is_model else "metabase/queries"

        # Save with all necessary attributes
        with open(f"{export_folder}/{query_name}", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "id": query_data["id"],
                    "name": query_data["name"],
                    "description": query_data.get("description", ""),
                    "database_id": query_data["dataset_query"]["database"],
                    "query": query_data["dataset_query"]["native"]["query"],
                    "dataset": is_model,
                    "type": "model" if is_model else "question",
                    "visualization_settings": query_data.get("visualization_settings", {}),
                    "result_metadata": query_data.get("result_metadata", []),
                    "collection_id": query_data.get("collection_id", None),
                    "display": query_data.get("display", "table"),
                    "parameter_mappings": query_data.get("parameter_mappings", []),
                    "parameters": query_data.get("parameters", []),
                },
                f,
                indent=4,
            )

print("Queries and models exported successfully.")
