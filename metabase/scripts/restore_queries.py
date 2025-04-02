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


# Fetch all existing queries and models separately
def get_existing_items():
    """Fetch all existing queries and models in Metabase."""
    response = requests.get(f"{METABASE_URL}/api/card", headers=metabase.headers)
    if response.status_code == 200:
        existing = {"queries": {}, "models": {}}
        for item in response.json():
            if item.get("dataset", False):  # True = Model
                existing["models"][item["name"]] = item["id"]
            else:  # False = Question
                existing["queries"][item["name"]] = item["id"]
        return existing
    else:
        print(f"Failed to fetch existing items: {response.text}")
        return {"queries": {}, "models": {}}


existing_items = get_existing_items()

# Paths for queries and models
queries_path = "metabase/queries/"
models_path = "metabase/models/"


def restore_card(query_file, is_model=False):
    """Restores either a query or a model, skipping if it already exists."""
    with open(query_file, "r", encoding="utf-8") as f:
        query_data = json.load(f)

    query_name = query_data["name"]

    # Check if the item already exists in the correct category
    if is_model and query_name in existing_items["models"]:
        print(f"⚠️ Skipping Model: {query_name} (already exists in Metabase)")
        return
    elif not is_model and query_name in existing_items["queries"]:
        print(f"⚠️ Skipping Query: {query_name} (already exists in Metabase)")
        return

    # Ensure `description` is `null` when missing or empty
    description = query_data.get("description", None)
    if description == "":
        description = None

    query_payload = {
        "name": query_name,
        "description": description,
        "dataset_query": {"type": "native", "native": {"query": query_data["query"]}, "database": query_data["database_id"]},
        "dataset": is_model,
        "type": "model" if is_model else "question",
        "display": query_data.get("display", "table"),
        "visualization_settings": query_data.get("visualization_settings", {}),
        "result_metadata": query_data.get("result_metadata", []),
        "collection_id": query_data.get("collection_id", None),
        "parameter_mappings": query_data.get("parameter_mappings", []),
        "parameters": query_data.get("parameters", []),
    }

    response = requests.post(f"{METABASE_URL}/api/card", headers=metabase.headers, json=query_payload)

    if response.status_code == 200:
        print(f"{'Model' if is_model else 'Query'} restored: {query_name}")
    else:
        print(f"Error restoring {query_name}: {response.text}")


# Restore queries
for query_file in os.listdir(queries_path):
    if query_file.endswith(".json"):
        restore_card(os.path.join(queries_path, query_file), is_model=False)

# Restore models
for model_file in os.listdir(models_path):
    if model_file.endswith(".json"):
        restore_card(os.path.join(models_path, model_file), is_model=True)
