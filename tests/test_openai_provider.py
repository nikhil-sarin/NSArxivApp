import os
import unittest
from unittest import mock

from app.summarizer import PaperSummarizer, _gemini_generation_config, _gemini_text


class OpenAIProviderTests(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {
            "SUMMARIZER_PROVIDER": "openai",
            "OPENAI_BASE_URL": "https://inference.handley-lab.co.uk/v1/",
            "OPENAI_API_KEY": "test-key",
            "LLM_MODEL": "google/gemma-4-31b-it",
        }, clear=False)
        self.env.start()

    def tearDown(self):
        self.env.stop()

    @staticmethod
    def response():
        response = mock.Mock()
        response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        return response

    @mock.patch("app.summarizer.requests.post")
    def test_summary_uses_configured_openai_compatible_endpoint(self, post):
        post.return_value = self.response()
        summarizer = PaperSummarizer()

        self.assertEqual(summarizer._call_openai("system", "paper", 8, False), "ok")
        self.assertEqual(post.call_args.args[0], "https://inference.handley-lab.co.uk/v1/chat/completions")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "google/gemma-4-31b-it")

    @mock.patch("app.summarizer.requests.post")
    def test_chat_uses_configured_openai_compatible_endpoint(self, post):
        post.return_value = self.response()
        summarizer = PaperSummarizer()

        self.assertEqual(summarizer._dispatch_chat("system", [{"role": "user", "content": "hello"}], provider="openai"), "ok")
        self.assertEqual(post.call_args.args[0], "https://inference.handley-lab.co.uk/v1/chat/completions")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "google/gemma-4-31b-it")

    def test_provider_specific_models_keep_private_ollama_separate(self):
        with mock.patch.dict(os.environ, {
            "SUMMARIZER_PROVIDER": "gemini",
            "GEMINI_MODEL": "gemini-2.5-pro",
            "LLM_MODEL": "google/gemma-4-31b-it",
            "OLLAMA_MODEL": "gemma4:latest",
        }, clear=False):
            self.assertEqual(
                PaperSummarizer(provider="gemini")._active_model(),
                "gemini-2.5-pro",
            )
            self.assertEqual(
                PaperSummarizer(provider="ollama").model,
                "gemma4:latest",
            )
            self.assertEqual(
                PaperSummarizer(provider="ollama")._active_provider(),
                "ollama",
            )

    def test_gemini_reserves_thinking_budget_and_extracts_visible_text(self):
        with mock.patch.dict(os.environ, {"GEMINI_THINKING_BUDGET": "128"}):
            config = _gemini_generation_config(16)
        self.assertEqual(config["maxOutputTokens"], 512)
        self.assertEqual(config["thinkingConfig"]["thinkingBudget"], 128)
        self.assertEqual(_gemini_text({
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
        }), "ok")
        with self.assertRaisesRegex(RuntimeError, "Gemini returned no text"):
            _gemini_text({
                "candidates": [{"content": {}, "finishReason": "MAX_TOKENS"}],
                "usageMetadata": {"thoughtsTokenCount": 16},
            })


if __name__ == "__main__":
    unittest.main()
