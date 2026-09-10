import unittest

from app.arxiv_client import ArxivClient


class ArxivClientMetadataTests(unittest.TestCase):
    def setUp(self):
        self.client = ArxivClient(download_dir="/tmp/nsarxivapp-test-papers")

    def test_extracts_current_two_cell_metadata(self):
        html = """
        <table>
          <tr>
            <td class="tablecell label">Comments:</td>
            <td class="tablecell comments mathjax">
              22 pages. Accepted for publication in The Astrophysical Journal
            </td>
          </tr>
          <tr>
            <td class="tablecell label">Journal reference:</td>
            <td class="tablecell jref">ApJ 999, 1 (2026)</td>
          </tr>
        </table>
        """

        self.assertEqual(
            self.client._extract_table_value(html, "Comments"),
            "22 pages. Accepted for publication in The Astrophysical Journal",
        )
        self.assertEqual(
            self.client._extract_table_value(html, "Journal reference"),
            "ApJ 999, 1 (2026)",
        )

    def test_extracts_legacy_inline_metadata(self):
        html = """
        <td><span class="descriptor">Comments:</span>
          Submitted to ApJ
        </td>
        """

        self.assertEqual(
            self.client._extract_table_value(html, "Comments"),
            "Submitted to ApJ",
        )


if __name__ == "__main__":
    unittest.main()
