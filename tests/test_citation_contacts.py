from unittest import mock

from app import citation_contacts


def test_explicit_corresponding_author_and_email_are_paired():
    tex = r"""
    \correspondingauthor{Yong-Feng Huang}
    \email{hyf@nju.edu.cn}
    \author{Xiao-Fei Dong}
    """
    assert citation_contacts.contact_from_tex(tex) == {
        "name": "Yong-Feng Huang",
        "email": "hyf@nju.edu.cn",
        "source": "arXiv source",
    }


def test_named_arxiv_source_contact_is_preferred_over_rendered_fallback():
    text = "Authors and affiliations\nContact: person@institute.edu\nAbstract"
    named = {
        "name": "Named Researcher",
        "email": "person@institute.edu",
        "source": "arXiv source",
    }
    with mock.patch.object(
        citation_contacts, "fetch_arxiv_contact", return_value=named
    ) as fetch:
        contact = citation_contacts.find_public_contact(text, "2607.23114")
    assert contact == named
    fetch.assert_called_once_with("2607.23114")


def test_rendered_email_is_retained_when_source_lookup_fails():
    text = "Authors and affiliations\nContact: person@institute.edu\nAbstract"
    with mock.patch.object(
        citation_contacts,
        "fetch_arxiv_contact",
        side_effect=OSError("offline"),
    ):
        contact = citation_contacts.find_public_contact(text, "2607.23114")
    assert contact["email"] == "person@institute.edu"
    assert citation_contacts.display_name(contact) == "Public contact"


def test_no_email_returns_none():
    assert citation_contacts.contact_from_text("No public contact here") is None


def test_mailto_url_contains_contact_and_paper_subject():
    url = citation_contacts.mailto_url(
        {"email": "person@institute.edu"}, "2609.08324"
    )
    assert url.startswith("mailto:person@institute.edu?")
    assert "subject=On+your+paper+%22arXiv%3A2609.08324%22" in url
