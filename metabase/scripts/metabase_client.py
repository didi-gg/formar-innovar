import requests
import os
import dotenv


class MetabaseClient:
    """Handles authentication and API requests to Metabase."""

    def __init__(self):
        """Load environment variables and authenticate to Metabase."""
        dotenv.load_dotenv()

        self.METABASE_URL = os.getenv("METABASE_URL")
        self.METABASE_USER = os.getenv("METABASE_USER")
        self.METABASE_PASSWORD = os.getenv("METABASE_PASSWORD")

        self.session_token = self.authenticate()
        self.headers = {"X-Metabase-Session": self.session_token, "Content-Type": "application/json"}

    def authenticate(self):
        """Authenticate with Metabase and return a session token."""
        session_res = requests.post(f"{self.METABASE_URL}/api/session", json={"username": self.METABASE_USER, "password": self.METABASE_PASSWORD})

        if session_res.status_code != 200:
            print(f"Failed to authenticate: {session_res.text}")
            exit(1)

        return session_res.json().get("id")
