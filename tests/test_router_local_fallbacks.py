import unittest
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


if __name__ == "__main__":
    unittest.main()
