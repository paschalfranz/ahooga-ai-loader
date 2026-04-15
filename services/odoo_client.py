import os
import requests
from dotenv import load_dotenv

load_dotenv()

ODOO_URL = os.getenv("ODOO_URL")
ODOO_DB = os.getenv("ODOO_DB")
ODOO_USERNAME = os.getenv("ODOO_USERNAME")
ODOO_API_KEY = os.getenv("ODOO_API_KEY")


class OdooClient:
    def __init__(self):
        self.url = ODOO_URL
        self.db = ODOO_DB
        self.username = ODOO_USERNAME
        self.api_key = ODOO_API_KEY
        self.uid = None

    def login(self):
        payload = {
            "jsonrpc": "2.0",
            "method": "call",
            "params": {
                "service": "common",
                "method": "login",
                "args": [self.db, self.username, self.api_key],
            },
            "id": 1,
        }

        response = requests.post(self.url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "result" not in data:
            raise Exception(f"Odoo login failed: {data}")

        self.uid = data["result"]
        return self.uid

    def execute_kw(self, model, method, args=None, kwargs=None):
        if self.uid is None:
            self.login()

        payload = {
            "jsonrpc": "2.0",
            "method": "call",
            "params": {
                "service": "object",
                "method": "execute_kw",
                "args": [
                    self.db,
                    self.uid,
                    self.api_key,
                    model,
                    method,
                    args or [],
                    kwargs or {},
                ],
            },
            "id": 2,
        }

        response = requests.post(self.url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        if "result" not in data:
            raise Exception(f"Odoo execute_kw failed: {data}")

        return data["result"]

    def search_read(self, model, domain=None, fields=None, limit=50, offset=0, order=None):
        kwargs = {
            "fields": fields or [],
            "limit": limit,
            "offset": offset,
        }
        if order:
            kwargs["order"] = order

        return self.execute_kw(
            model=model,
            method="search_read",
            args=[domain or []],
            kwargs=kwargs,
        )

    def read(self, model, ids, fields=None):
        return self.execute_kw(
            model=model,
            method="read",
            args=[ids],
            kwargs={"fields": fields or []},
        )