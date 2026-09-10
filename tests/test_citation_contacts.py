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


def test_public_email_in_rendered_front_matter_is_used_without_network():
    text = "Authors and affiliations\nContact: person@institute.edu\nAbstract"
    with mock.patch.object(citation_contacts, "fetch_arxiv_contact") as fetch:
        contact = citation_contacts.find_public_contact(text, "2607.23114")
    assert contact["email"] == "person@institute.edu"
    fetch.assert_not_called()


def test_no_email_returns_none():
    assert citation_contacts.contact_from_text("No public contact here") is None
