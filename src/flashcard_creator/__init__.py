"""flashcard-creator: an Open Notebook creator that turns notebook content into
spaced-repetition flashcards (emitted as ``flashcards.v1``) plus a downloadable
Anki ``.apkg`` deck.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import uuid
from importlib import resources
from pathlib import Path
from typing import ClassVar, List

from ai_prompter import Prompter
from loguru import logger
from open_notebook_creator_sdk import (
    BaseCreator,
    CreationError,
    CreationFile,
    CreationRequest,
    CreationResult,
    CreatorManifest,
    ModelRoleSpec,
)
from open_notebook_creator_sdk.schemas import FlashcardsV1
from pydantic import BaseModel, Field

__version__ = "0.1.0"


class FlashcardsConfig(BaseModel):
    """Per-generation config; drives the host's generate form."""

    num_cards: int = Field(default=15, ge=1, le=100, description="How many cards to generate")
    deck_name: str = Field(default="Open Notebook Deck", description="Anki deck name")


def _stable_id(seed: str) -> int:
    """genanki wants stable 32-bit-ish ints so re-imports update, not duplicate."""
    return int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()


def _build_apkg(deck_name: str, cards: List[dict], seed: str, out_path: Path) -> None:
    import genanki

    model = genanki.Model(
        _stable_id(f"model:{seed}"),
        "Open Notebook Basic",
        fields=[{"name": "Front"}, {"name": "Back"}],
        templates=[
            {
                "name": "Card 1",
                "qfmt": "{{Front}}",
                "afmt": '{{FrontSide}}<hr id="answer">{{Back}}',
            }
        ],
    )
    deck = genanki.Deck(_stable_id(f"deck:{seed}"), deck_name)
    for c in cards:
        tags = [re.sub(r"\s+", "_", t) for t in c.get("tags", [])]
        deck.add_note(genanki.Note(model=model, fields=[c["front"], c["back"]], tags=tags))
    genanki.Package(deck).write_to_file(str(out_path))


class FlashcardCreator(BaseCreator):
    config_model: ClassVar[type] = FlashcardsConfig

    @property
    def manifest(self) -> CreatorManifest:
        return self.build_manifest(
            key="flashcards",
            name="Flashcards",
            version=__version__,
            description="LLM-generated Q/A cards for in-app study and Anki export.",
            sdk_compat=">=0.1,<1",
            emits=["flashcards.v1"],
            model_roles=[
                ModelRoleSpec(
                    key="text",
                    kind="language",
                    requires=["structured_json"],
                    description="LLM that writes the flashcards.",
                )
            ],
            icon="layers",
        )

    async def generate(self, request: CreationRequest) -> CreationResult:
        cfg = FlashcardsConfig.model_validate(request.config)
        role = request.models.get("text")
        if role is None:
            return CreationResult(
                status="FAILURE",
                schema_id="flashcards.v1",
                data={},
                errors=[CreationError(phase="setup", message="missing 'text' model role")],
                user_message="No language model was provided for flashcard generation.",
            )

        # 1. LLM -> cards JSON
        template = resources.files("flashcard_creator.prompts").joinpath(
            "flashcards.jinja"
        ).read_text()
        prompt = Prompter(template_text=template).render(
            {
                "content": request.content.text,
                "num_cards": cfg.num_cards,
                "deck_name": cfg.deck_name,
                "language": request.language,
            }
        )
        llm = role.create_language(structured={"type": "json"}, max_tokens=4000)
        resp = await llm.ainvoke(prompt)
        raw = resp.content if hasattr(resp, "content") else str(resp)
        try:
            parsed = json.loads(_strip_fences(raw))
        except json.JSONDecodeError as e:
            logger.error(f"flashcards: model returned non-JSON: {e}")
            return CreationResult(
                status="FAILURE",
                schema_id="flashcards.v1",
                data={},
                errors=[CreationError(phase="parse", message=f"invalid JSON: {e}", retryable=True)],
                user_message="The model returned an unparseable response. Please retry.",
            )

        cards_in = parsed.get("cards", []) if isinstance(parsed, dict) else []
        cards = []
        for c in cards_in:
            front = (c.get("front") or "").strip()
            back = (c.get("back") or "").strip()
            if not front or not back:
                continue
            cards.append(
                {"id": str(uuid.uuid4()), "front": front, "back": back, "tags": c.get("tags", []) or []}
            )

        if not cards:
            return CreationResult(
                status="FAILURE",
                schema_id="flashcards.v1",
                data={},
                errors=[CreationError(phase="generate", message="no valid cards produced")],
                user_message="No flashcards could be generated from this content.",
            )

        data = FlashcardsV1(deck_name=cfg.deck_name, cards=cards).model_dump()

        # 2. Build .apkg (best-effort -> PARTIAL on failure)
        files: list[CreationFile] = []
        warnings: list[str] = []
        errors: list[CreationError] = []
        rel_name = f"{re.sub(r'[^A-Za-z0-9._-]+', '_', cfg.deck_name) or 'deck'}.apkg"
        out_path = Path(request.output_dir) / rel_name
        try:
            await asyncio.to_thread(
                _build_apkg, cfg.deck_name, cards, request.artifact_id, out_path
            )
            files.append(
                CreationFile(
                    filename=rel_name,
                    content_type="application/octet-stream",
                    path=rel_name,
                    label="anki_deck",
                )
            )
        except Exception as e:  # noqa: BLE001 - packaging is non-fatal
            logger.warning(f"flashcards: .apkg build failed: {e}")
            warnings.append("Anki .apkg export failed; cards are still available for in-app study.")
            errors.append(CreationError(phase="apkg", message=str(e)))

        return CreationResult(
            status="PARTIAL" if errors else "SUCCESS",
            schema_id="flashcards.v1",
            data=data,
            files=files,
            warnings=warnings,
            errors=errors,
        )
