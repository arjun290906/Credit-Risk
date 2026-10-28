"""Integration test for FastAPI server running over HTTP."""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# Ensure paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
MODEL_DIR = PROJECT_ROOT / "Model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import uvicorn
from main import app


class ServerThread(threading.Thread):
    def __init__(self, host: str = "127.0.0.1", port: int = 8008):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning")
        self.server = uvicorn.Server(config)

    def run(self):
        self.server.run()

    def stop(self):
        self.server.should_exit = True


def test_http_api():
    server = ServerThread(port=8008)
    server.start()
    time.sleep(2)  # Give server time to bind

    base_url = "http://127.0.0.1:8008"

    try:
        # 1. Test /health
        print("1. Testing GET /health...")
        req = urllib.request.Request(f"{base_url}/health")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            print("   /health response:", data)
            assert data["status"] == "healthy"
            assert "risk_thresholds" in data

        # 2. Test /predict
        print("\n2. Testing POST /predict...")
        payload = {
            "person_age": 30,
            "person_income": 60000,
            "person_home_ownership": "MORTGAGE",
            "person_emp_length": 4.0,
            "credit_history_length": 6,
            "loan_intent": "PERSONAL",
            "loan_grade": "B",
            "loan_amnt": 8000,
            "loan_int_rate": 9.5,
            "loan_percent_income": 0.13,
        }
        json_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/predict",
            data=json_data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            print(f"   /predict response: p_default={data['p_default']}, risk_category={data['risk_category']}")
            print(f"   Top reason: {data['reasons'][0]['plain_text']}")
            print(f"   Explanation check valid: {data['explanation_check']['is_valid']}")
            assert "p_default" in data
            assert data["risk_category"] in ["Low", "Medium", "High", "Very High"]
            assert len(data["reasons"]) == 5
            assert data["explanation_check"]["is_valid"] is True

        # 3. Test /shap/global
        print("\n3. Testing GET /shap/global...")
        req = urllib.request.Request(f"{base_url}/shap/global?grouped=true")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            print(f"   Global features returned: {len(data)}")
            print(f"   Top global driver: {data[0]['feature']} ({data[0]['importance_pct']:.1f}%)")
            assert len(data) >= 8

        print("\nALL HTTP API TESTS PASSED SUCCESSFULLY!")

    finally:
        server.stop()


if __name__ == "__main__":
    test_http_api()
