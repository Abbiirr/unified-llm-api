import unittest
from unittest import mock
from pathlib import Path

import yaml

import smart_router


ROOT = Path(__file__).resolve().parents[1]


class RouterRescuePolicyTests(unittest.TestCase):
    def test_local_only_aliases_are_detected(self):
        self.assertTrue(smart_router.is_local_only_alias("local"))
        self.assertTrue(smart_router.is_local_only_alias("tools_local"))
        self.assertTrue(smart_router.is_local_only_alias("llama_local"))
        self.assertTrue(smart_router.is_local_only_alias("ollama_chat/qwen3.5:9b"))
        self.assertFalse(smart_router.is_local_only_alias("default"))

    def test_llama_local_timeout_rescue_stays_local(self):
        self.assertEqual(smart_router.pick_408_rescue_aliases("llama_local"), ("local",))

    def test_explicit_local_aliases_do_not_escape_to_cloud_on_timeout(self):
        self.assertEqual(smart_router.pick_408_rescue_aliases("local"), ())
        self.assertEqual(smart_router.pick_408_rescue_aliases("tools_local"), ())

    def test_default_timeout_rescue_keeps_existing_cloud_chain(self):
        self.assertEqual(
            smart_router.pick_408_rescue_aliases("default"),
            ("big", "default_cloud"),
        )

    def test_lmw_wrapper_probe_uses_configured_api_key(self):
        with mock.patch.dict(
            smart_router.os.environ,
            {"LMW_API_KEY": "lmw-test-key"},
            clear=False,
        ):
            self.assertEqual(
                smart_router.ollama_probe_headers(
                    "OLLAMA_HOST_3",
                    "http://100.111.28.50:11435",
                ),
                {"x-api-key": "lmw-test-key"},
            )

    def test_plain_ollama_probe_does_not_send_lmw_key(self):
        with mock.patch.dict(
            smart_router.os.environ,
            {"LMW_API_KEY": "lmw-test-key"},
            clear=False,
        ):
            self.assertEqual(
                smart_router.ollama_probe_headers(
                    "OLLAMA_HOST_1",
                    "http://10.112.30.10:11434",
                ),
                {},
            )


class LiteLLMFallbackConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "litellm_config.yaml").open() as handle:
            cls.config = yaml.safe_load(handle)
        cls.fallbacks = {
            next(iter(entry)): next(iter(entry.values()))
            for entry in cls.config["litellm_settings"]["fallbacks"]
        }

    def test_default_has_a_declared_fallback_chain(self):
        self.assertEqual(self.fallbacks["default"], ["default_cloud", "tools_local"])

    def test_llama_local_fallbacks_remain_local(self):
        self.assertEqual(self.fallbacks["llama_local"], ["local", "tools_local"])

    def test_qwen36_h1_is_wired_into_coding_and_reasoning_aliases(self):
        h1_aliases = {
            entry["model_name"]
            for entry in self.config["model_list"]
            if entry.get("litellm_params", {}).get("model") == "ollama_chat/qwen3.6:27b"
            and entry.get("litellm_params", {}).get("api_base") == "os.environ/OLLAMA_HOST_1"
        }

        for alias in {"coding", "default", "thinking", "big"}:
            self.assertIn(alias, h1_aliases)

        coding_h1_entries = [
            entry
            for entry in self.config["model_list"]
            if entry["model_name"] == "coding"
            and entry.get("litellm_params", {}).get("model") == "ollama_chat/qwen3.6:27b"
            and entry.get("litellm_params", {}).get("api_base") == "os.environ/OLLAMA_HOST_1"
        ]
        self.assertTrue(coding_h1_entries)
        self.assertTrue(
            all(
                entry.get("model_info", {}).get("supports_function_calling") is True
                for entry in coding_h1_entries
            )
        )

    def test_unified_ocr_alias_prefers_qwen36_h1_for_fast_multimodal_routing(self):
        alias_entries = [
            entry for entry in self.config["model_list"] if entry["model_name"] == "ocr"
        ]
        self.assertTrue(alias_entries)

        preferred = min(
            alias_entries,
            key=lambda entry: entry.get("litellm_params", {}).get("order", 999),
        )

        self.assertEqual(
            preferred.get("litellm_params", {}).get("model"),
            "ollama_chat/qwen3.6:27b",
        )
        self.assertEqual(
            preferred.get("litellm_params", {}).get("api_base"),
            "os.environ/OLLAMA_HOST_1",
        )
        self.assertEqual(preferred.get("litellm_params", {}).get("order"), 1)


if __name__ == "__main__":
    unittest.main()
