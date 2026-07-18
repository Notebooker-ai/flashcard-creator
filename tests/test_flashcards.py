"""Tests for FlashcardCreator using a stubbed language model (no network)."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import pytest
from flashcard_creator import FlashcardCreator
from open_notebook_creator_sdk import ContentBundle, CreationRequest, ModelRole
from open_notebook_creator_sdk.testing import assert_creator_compliant, assert_result_compliant


class _FakeResp:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, payload: str):
        self._payload = payload

    async def ainvoke(self, _prompt):
        return _FakeResp(self._payload)


class _FakeRole(ModelRole):
    """A ModelRole whose create_language returns a canned LLM."""

    payload: str = ""

    def create_language(self, **_):
        return _FakeLLM(self.payload)


def _role(cards):
    return _FakeRole(
        provider="fake",
        model="fake",
        payload=json.dumps({"deck_name": "Deck", "cards": cards}),
    )


def test_static_compliance():
    assert_creator_compliant(FlashcardCreator())


@pytest.mark.asyncio
async def test_generate_produces_cards_and_apkg():
    creator = FlashcardCreator()
    with tempfile.TemporaryDirectory() as td:
        req = CreationRequest(
            content=ContentBundle(text="Photosynthesis converts light to energy."),
            config={"num_cards": 2, "deck_name": "Bio"},
            models={"text": _role([
                {"front": "What does photosynthesis do?", "back": "Converts light to energy", "tags": ["bio"]},
                {"front": "What organelle performs it?", "back": "Chloroplast"},
            ])},
            output_dir=td,
            artifact_id="art-1",
        )
        result = await creator.generate(req)

        assert result.status == "SUCCESS"
        assert_result_compliant(creator, result)
        assert len(result.data["cards"]) == 2
        assert result.data["cards"][0]["id"]  # ids assigned

        # .apkg exists, is contained, and is a valid zip (Anki package = zip)
        assert len(result.files) == 1
        apkg = Path(td) / result.files[0].path
        assert apkg.exists()
        assert zipfile.is_zipfile(apkg)


@pytest.mark.asyncio
async def test_invalid_json_is_failure():
    creator = FlashcardCreator()
    with tempfile.TemporaryDirectory() as td:
        role = _FakeRole(provider="f", model="f", payload="not json at all")
        req = CreationRequest(
            content=ContentBundle(text="x"),
            models={"text": role},
            output_dir=td,
            artifact_id="a",
        )
        result = await creator.generate(req)
        assert result.status == "FAILURE"
        assert result.errors[0].phase == "parse"


@pytest.mark.asyncio
async def test_strips_markdown_fences():
    creator = FlashcardCreator()
    fenced = "```json\n" + json.dumps(
        {"deck_name": "D", "cards": [{"front": "Q", "back": "A"}]}
    ) + "\n```"
    with tempfile.TemporaryDirectory() as td:
        req = CreationRequest(
            content=ContentBundle(text="x"),
            config={"deck_name": "D"},
            models={"text": _FakeRole(provider="f", model="f", payload=fenced)},
            output_dir=td,
            artifact_id="a",
        )
        result = await creator.generate(req)
        assert result.status == "SUCCESS"
        assert len(result.data["cards"]) == 1


@pytest.mark.asyncio
async def test_skips_incomplete_cards():
    creator = FlashcardCreator()
    with tempfile.TemporaryDirectory() as td:
        req = CreationRequest(
            content=ContentBundle(text="x"),
            models={"text": _role([
                {"front": "Q", "back": "A"},
                {"front": "", "back": "B"},
                {"front": "C", "back": ""},
            ])},
            output_dir=td,
            artifact_id="a",
        )
        result = await creator.generate(req)
        assert result.status == "SUCCESS"
        assert len(result.data["cards"]) == 1  # incomplete cards dropped


@pytest.mark.asyncio
async def test_no_text_role_is_failure():
    creator = FlashcardCreator()
    with tempfile.TemporaryDirectory() as td:
        req = CreationRequest(content=ContentBundle(text="x"), output_dir=td, artifact_id="a")
        result = await creator.generate(req)
        assert result.status == "FAILURE"
        assert result.errors[0].phase == "setup"


def test_manifest_declares_view_bundle_and_it_ships():
    """The creator owns its UI: the manifest points at a shipped HTML view bundle."""
    from importlib import resources

    m = FlashcardCreator().manifest
    assert m.view is not None
    assert m.view.entry == "view/index.html"
    asset = resources.files("flashcard_creator").joinpath(m.view.entry)
    assert asset.is_file()
    html = asset.read_text()
    # self-contained + speaks the host handshake + dispatches our schema
    assert "open-notebook:ready" in html
    assert "open-notebook:artifact" in html
    assert "flashcards.v1" in html
    assert "<script src" not in html  # no external scripts (sandbox-safe, offline)


def test_view_bundle_speaks_the_review_contract():
    """The study UI persists spaced-repetition state through the host: the view
    must consume `review` from the artifact message and emit review-update
    messages, using ts-fsrs-shaped card state."""
    from importlib import resources

    html = (
        resources.files("flashcard_creator").joinpath("view/index.html").read_text()
    )
    # Host -> view: saved review state rides in on the artifact message.
    assert "open-notebook:artifact" in html
    assert "msg.review" in html
    # View -> host: every grade is posted back for persistence.
    assert "open-notebook:review-update" in html
    # ts-fsrs Card shape keeps the stored state interoperable.
    for field in ("stability", "difficulty", "scheduled_days", "last_review", "lapses"):
        assert f'"{field}"' in html or f"{field}:" in html
