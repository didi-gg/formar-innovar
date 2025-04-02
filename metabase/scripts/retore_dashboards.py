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

# TODO