import json
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from fastapi.testclient import TestClient

import smart_router


async def _noop_async():
    return None


class DummyResponse:
    def __init__(self, status_code=200, content=None, headers=None):
        self.status_code = status_code
        self.content = content or b""
        self.headers = headers or {"content-type": "application/json"}


class RecordingClient:
    def __init__(self, response=None):
        self.requests = []
        self.response = response or DummyResponse(
            status_code=200,
            content=json.dumps(
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            ).encode(),
            headers={
                "content-type": "application/json",
                "x-litellm-model-api-base": "http://example.invalid",
                "x-litellm-model-id": "test-model",
            },
        )

    async def request(self, method, url, headers=None, content=None):
        self.requests.append(
            {
                "method": method,
                "url": url,
                "headers": headers or {},
                "content": content or b"",
            }
        )
        return self.response

    async def send(self, request, stream=False):
        raise AssertionError("streaming was not expected in this test")

    async def aclose(self):
        return None


@contextmanager
def router_client():
    original_http_client = smart_router.http_client
    with patch.object(smart_router, "_build_model_identity_map", new=_noop_async), patch.object(
        smart_router, "probe_ollama_hosts", new=_noop_async
    ):
        try:
            with TestClient(smart_router.app) as client:
                yield client
        finally:
            smart_router.http_client = original_http_client


class PublicRouterEndpointTests(unittest.TestCase):
    def test_models_endpoint_is_public_and_lists_aliases(self):
        with router_client() as client:
            response = client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        ids = {item["id"] for item in payload["data"]}
        self.assertIn("default", ids)
        self.assertIn("coding", ids)
        self.assertIn("tools", ids)

    def test_invalid_chat_body_returns_clear_400(self):
        with router_client() as client:
            response = client.post(
                "/v1/chat/completions",
                content="{not-json}",
                headers={"Content-Type": "application/json"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid JSON request body", response.text)

    def test_null_model_is_defaulted_before_forwarding(self):
        fake_client = RecordingClient()

        with router_client() as client:
            smart_router.http_client = fake_client
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": None,
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 5,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(fake_client.requests), 1)
        forwarded = json.loads(fake_client.requests[0]["content"])
        self.assertTrue(forwarded["model"])

    def test_ocr_alias_stays_unified_before_forwarding(self):
        fake_client = RecordingClient()

        with router_client() as client:
            smart_router.http_client = fake_client
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "ocr",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 5,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(fake_client.requests), 1)
        forwarded = json.loads(fake_client.requests[0]["content"])
        self.assertEqual(forwarded["model"], "ocr")


if __name__ == "__main__":
    unittest.main()
