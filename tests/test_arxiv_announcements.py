from datetime import date
from unittest import mock

import requests

from app import arxiv_announcements


def _paper(arxiv_id: str) -> arxiv_announcements.AnnouncementPaper:
    return arxiv_announcements.AnnouncementPaper(
        arxiv_id=arxiv_id,
        title="Paper",
        authors=["A. Author"],
        abstract="Abstract",
        primary_category="astro-ph.HE",
        categories=["astro-ph.HE"],
        announced_date="2026-10-30",
        v1_date="2026-10-29",
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
    )


def test_catchup_switches_batch_to_abs_pages_after_oai_timeout():
    ids = ["2610.41005", "2610.41006"]
    with mock.patch.object(
        arxiv_announcements, "fetch_catchup_ids", return_value=ids
    ), mock.patch.object(
        arxiv_announcements,
        "fetch_record",
        side_effect=requests.Timeout("OAI timed out"),
    ) as oai, mock.patch.object(
        arxiv_announcements,
        "fetch_record_from_abs_page",
        side_effect=[_paper(ids[0]), _paper(ids[1])],
    ) as fallback, mock.patch.object(arxiv_announcements.time, "sleep"):
        papers = arxiv_announcements.fetch_via_catchup(
            "astro-ph.HE", date(2026, 10, 30)
        )

    assert [paper.arxiv_id for paper in papers] == ids
    oai.assert_called_once_with(ids[0])
    assert fallback.call_count == 2
