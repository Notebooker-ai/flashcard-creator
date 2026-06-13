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
