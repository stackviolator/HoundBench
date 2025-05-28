import hmac
import hashlib
import base64
import requests
import datetime
import json
import os
from dotenv import load_dotenv

from typing import Optional

# Load environment variables
load_dotenv()

# Replace hardcoded constants with environment variables
BHE_DOMAIN = os.getenv("BHE_DOMAIN")
BHE_PORT = int(os.getenv("BHE_PORT", "443"))
BHE_SCHEME = os.getenv("BHE_SCHEME", "https")
BHE_TOKEN_ID = os.getenv("BHE_TOKEN_ID")
BHE_TOKEN_KEY = os.getenv("BHE_TOKEN_KEY")

PRINT_PRINCIPALS = os.getenv("PRINT_PRINCIPALS", "false").lower() == "true"
PRINT_ATTACK_PATH_TIMELINE_DATA = os.getenv("PRINT_ATTACK_PATH_TIMELINE_DATA", "false").lower() == "true"
PRINT_POSTURE_DATA = os.getenv("PRINT_POSTURE_DATA", "false").lower() == "true"

DATA_START = "1970-01-01T00:00:00.000Z"
DATA_END = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z' # Now

class Credentials(object):
    def __init__(self, token_id: str, token_key: str) -> None:
        self.token_id = token_id
        self.token_key = token_key


class APIVersion(object):
    def __init__(self, api_version: str, server_version: str) -> None:
        self.api_version = api_version
        self.server_version = server_version


class Domain(object):
    def __init__(self, name: str, id: str, collected: bool, domain_type: str) -> None:
        self.name = name
        self.id = id
        self.type = domain_type
        self.collected = collected

    def to_dict(self):
        return {
            "name": self.name,
            "id": self.id,
            "type": self.type,
            "collected": self.collected
        }

class Client(object):
    def __init__(self, scheme: str, host: str, port: int, credentials: Credentials) -> None:
        self._scheme = scheme
        self._host = host
        self._port = port
        self._credentials = credentials
        if not self.test_login():
            raise ValueError("Could not connect to BloodHound Instance. Please verify your API credentials and ensure the server is accessible.")

    # Test if the API tokens are valid
    def test_login(self) -> bool:
        response = self._request("GET", "/api/v2/tokens")
        if response.status_code == 200:
            return True
        else:
            return False

    def _format_url(self, uri: str) -> str:
        formatted_uri = uri
        if uri.startswith("/"):
            formatted_uri = formatted_uri[1:]

        return f"{self._scheme}://{self._host}:{self._port}/{formatted_uri}"

    # Taken from https://support.bloodhoundenterprise.io/hc/en-us/articles/11311053342619-Working-with-the-BloodHound-API
    def _request(self, method: str, uri: str, body: Optional[bytes] = None) -> requests.Response:
        digester = hmac.new(self._credentials.token_key.encode(), None, hashlib.sha256)

        digester.update(f"{method}{uri}".encode())

        digester = hmac.new(digester.digest(), None, hashlib.sha256)
        datetime_formatted = datetime.datetime.now().astimezone().isoformat("T")
        digester.update(datetime_formatted[:13].encode())
        digester = hmac.new(digester.digest(), None, hashlib.sha256)
        if body is not None:
            digester.update(body)

        return requests.request(
            method=method,
            url=self._format_url(uri),
            headers={
                "User-Agent": "bhe-python-sdk 0001",
                "Authorization": f"bhesignature {self._credentials.token_id}",
                "RequestDate": datetime_formatted,
                "Signature": base64.b64encode(digester.digest()),
                "Content-Type": "application/json",
            },
            data=body,
        )

    def get_version(self) -> APIVersion:
        response = self._request("GET", "/api/version")
        payload = response.json()

        return APIVersion(api_version=payload["data"]["API"]["current_version"], server_version=payload["data"]["server_version"])

    def get_domains(self) -> list[Domain]:
        response = self._request('GET', '/api/v2/available-domains')
        payload = response.json()['data']

        domains = list()
        for domain in payload:
            domains.append(Domain(domain["name"], domain["id"], domain["collected"], domain["type"]))

        return domains

    def get_users(self) -> list:
        response = self._request('GET', '/api/v2/users')
        payload = response.json()['data']

        users = list()
        for user in payload:
            # users.append(User(user["name"], user["id"], user["domain"], user["type"]))
            users.append(user)
        return users

    def run_cypher(self, query, include_properties=False) -> dict:
        """Runs a Cypher query and returns the results

        Parameters:
        query (string): The Cypher query to run
        include_properties (bool): Should all properties of result nodes/edges be returned

        Returns:
        dict: Response containing either data or error information
        """
        data = {
            "include_properties": include_properties,
            "query": query
        }
        body = json.dumps(data).encode('utf8')
        response = self._request("POST", "/api/v2/graphs/cypher", body)

        # Check for HTTP errors before attempting to parse JSON
        if response.status_code == 429:
            return {"error": "Rate limit exceeded (HTTP 429)", "details": "Too many requests. Please wait and try again later."}
        if response.status_code == 401:
            return {"error": "Unauthorized (HTTP 401)", "details": "Request not authorized. Check API token and permissions."}
        if response.status_code == 403:
            return {"error": "Forbidden (HTTP 403)", "details": "Access to the resource is forbidden."}
        if response.status_code == 500:
             return {"error": "Internal Server Error (HTTP 500)", "details": "The server encountered an internal error."}
        # You can add more specific status code checks here if needed
        
        # If status code is not 200 (OK) and not specifically handled above, return a generic HTTP error
        if response.status_code != 200:
            return {
                "error": f"HTTP Error: {response.status_code}",
                "details": response.text[:500] # Include some of the response text for context
            }

        try:
            response_json = response.json()
        except json.JSONDecodeError as e:
            return {
                "error": "JSONDecodeError",
                "details": f"Failed to parse API response as JSON. Status: {response.status_code}. Error: {e}. Response text (first 200 chars): {response.text[:200]}"
            }

        if 'errors' not in response_json:
            return response_json

        errors = response_json['errors']
        if any(error['message'] == "resource not found" for error in errors):
            return {"error": f"No matches found for query '{query}'"}

        error_messages = '\n'.join(f"'{error['message']}'" for error in errors)
        return {
            "error": f"Error in query '{query}'. Please fix and try again.",
            "details": error_messages
        }

def main() -> None:
    import sys
    credentials = Credentials(
        token_id=BHE_TOKEN_ID,
        token_key=BHE_TOKEN_KEY,
    )

    client = Client(scheme=BHE_SCHEME, host=BHE_DOMAIN, port=BHE_PORT, credentials=credentials)
    if not client.test_login():
        print("Login unsuccessful")
        sys.exit(1)

    print(sys.argv[1])

if __name__ == "__main__":
    main()
