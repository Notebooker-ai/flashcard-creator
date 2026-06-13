# flashcard-creator

An [Open Notebook](https://open-notebook.ai) **creator** plugin: turns notebook
content into spaced-repetition flashcards.

- Emits the `flashcards.v1` artifact schema (studied in-app with `ts-fsrs`).
- Exports a downloadable Anki `.apkg` deck via [`genanki`](https://github.com/kerrickstaley/genanki).
- Implements the [`open-notebook-creator-sdk`](https://github.com/Notebooker-ai/open-notebook-creator-sdk) `BaseCreator` contract and registers under the `open_notebook.creators` entry point.

## Model roles

| role | kind | requires |
|------|------|----------|
| `text` | language | `structured_json` |

## Config

| field | default | notes |
|-------|---------|-------|
| `num_cards` | 15 | 1–100 |
| `deck_name` | "Open Notebook Deck" | Anki deck name |

## Dev

```bash
uv sync --extra dev
uv run pytest
```

MIT licensed.
