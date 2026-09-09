import os
import unittest
from unittest import mock

from app.summarizer import PaperSummarizer


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


if __name__ == "__main__":
    unittest.main()
