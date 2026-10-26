import pytest

from ml_training.board_meeting_protocol.utils.backend import TextModel
from ml_training.board_meeting_protocol.utils.writer import ProtocolWriter


class FakeModel(TextModel):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def complete(self, prompt: str, model: str | None, on_delta=None) -> str:
        self.calls.append((prompt, model))
        if "Conclusions from earlier rows" in prompt:
            return "Finished protocol text"
        return "Row summary"


def test_predict_keeps_row_note_off_the_master_prompt():
    writer = ProtocolWriter().fit()
    backend = FakeModel()
    body = writer.predict(
        backend,
        mode="writing",
        rows=[
            {
                "left": "raw notes",
                "middle": "finished protocol",
                "right": "",
                "note": "mention the budget vote",
            }
        ],
        new_text="today's raw notes",
        orchestrator_note="keep it short",
        row_model="row-model",
        orchestrator_model="master-model",
    )
    assert body == "Finished protocol text"
    row_prompt, row_model = backend.calls[0]
    master_prompt, master_model = backend.calls[1]
    assert "mention the budget vote" in row_prompt
    assert "mention the budget vote" not in master_prompt
    assert "keep it short" in master_prompt
    assert "Raw notes:" in row_prompt
    assert "Finished protocol:" in row_prompt
    assert "carried from meeting to meeting" in row_prompt
    assert "one pass" in row_prompt
    assert "A second pass has no purpose" in row_prompt
    assert "own headings" in row_prompt
    assert "may not apply" in master_prompt
    assert "A second pass has no purpose" in master_prompt
    assert "HIGH UNCERTAINTY" in row_prompt
    assert "HIGH UNCERTAINTY:" in row_prompt
    assert "HIGH UNCERTAINTY" in master_prompt
    assert "HIGH UNCERTAINTY:" in master_prompt
    assert row_model == "row-model"
    assert master_model == "master-model"


def test_empty_notes_are_omitted():
    writer = ProtocolWriter().fit()
    backend = FakeModel()
    writer.predict(
        backend,
        mode="reviewing",
        rows=[{"left": "draft", "middle": "final", "right": "email"}],
        new_text="new draft",
    )
    row_prompt, _ = backend.calls[0]
    master_prompt, _ = backend.calls[1]
    assert "Additional context" not in row_prompt
    assert "Additional context" not in master_prompt
    assert "Feedback email:" in row_prompt
    assert "pedantic" in row_prompt
    assert "date the meeting was held" in row_prompt
    assert "may not apply" in master_prompt
    assert "A second pass has no purpose" in master_prompt
    assert "HIGH UNCERTAINTY" in master_prompt
    assert "HIGH UNCERTAINTY:" in master_prompt


def test_orchestrator_is_called_once_for_three_rows():
    writer = ProtocolWriter().fit()
    backend = FakeModel()
    writer.predict(
        backend,
        mode="writing",
        rows=[
            {"left": "raw one", "middle": "done one"},
            {"left": "raw two", "middle": "done two"},
            {"left": "", "middle": "", "right": "skip me"},
        ],
        new_text="today",
        row_model="row-model",
        orchestrator_model="master-model",
    )
    assert [model for _, model in backend.calls] == [
        "row-model",
        "row-model",
        "master-model",
    ]


def test_progress_shows_row_text_sent_to_the_orchestrator():
    writer = ProtocolWriter().fit()
    backend = FakeModel()
    events: list[dict] = []
    writer.predict(
        backend,
        mode="writing",
        rows=[{"left": "raw", "middle": "done"}],
        new_text="today",
        progress=events.append,
    )
    start = next(event for event in events if event["step"] == "orchestrator")
    assert "Row summary" in start["received"]
    assert events[-1]["reply"] == "Finished protocol text"


class BlankOrchestratorModel(FakeModel):
    def complete(self, prompt: str, model: str | None, on_delta=None) -> str:
        reply = super().complete(prompt, model, on_delta)
        if "Conclusions from earlier rows" in prompt:
            return "  \n"
        return reply


def test_unknown_mode_is_rejected():
    writer = ProtocolWriter().fit()
    backend = FakeModel()
    with pytest.raises(ValueError):
        writer.predict(backend, mode="summarizing", rows=[], new_text="today")
    assert backend.calls == []


def test_blank_new_text_is_rejected():
    writer = ProtocolWriter().fit()
    backend = FakeModel()
    with pytest.raises(ValueError):
        writer.predict(backend, mode="writing", rows=[], new_text="  \n\t")
    assert backend.calls == []


def test_only_the_first_three_complete_rows_are_used():
    writer = ProtocolWriter().fit()
    backend = FakeModel()
    writer.predict(
        backend,
        mode="writing",
        rows=[
            {"left": "raw zero", "middle": ""},
            {"left": "raw one", "middle": "done one"},
            {"left": "raw two", "middle": "done two"},
            {"left": "raw three", "middle": "done three"},
            {"left": "raw four", "middle": "done four"},
        ],
        new_text="today",
    )
    row_prompts = [prompt for prompt, _ in backend.calls[:-1]]
    assert len(row_prompts) == 3
    assert "raw one" in row_prompts[0]
    assert "raw two" in row_prompts[1]
    assert "raw three" in row_prompts[2]
    assert not any("raw zero" in prompt for prompt, _ in backend.calls)
    assert not any("raw four" in prompt for prompt, _ in backend.calls)


def test_blank_orchestrator_reply_raises():
    writer = ProtocolWriter().fit()
    backend = BlankOrchestratorModel()
    with pytest.raises(RuntimeError):
        writer.predict(
            backend,
            mode="reviewing",
            rows=[{"left": "draft", "middle": "final"}],
            new_text="new draft",
        )
