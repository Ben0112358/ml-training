from importlib.resources import files

from ml_training.board_meeting_protocol.utils.backend import TextModel

MAX_ROWS = 3


def _read_prompt(name: str) -> str:
    return (
        files("ml_training.board_meeting_protocol.prompts")
        .joinpath(name)
        .read_text(encoding="utf-8")
        .strip()
    )


class ProtocolWriter:
    """Prompt pack. fit() stores the briefs; predict() calls a text model."""

    def fit(self) -> "ProtocolWriter":
        self.row_brief = _read_prompt("row.md")
        self.master_writing = _read_prompt("master_writing.md")
        self.master_reviewing = _read_prompt("master_reviewing.md")
        return self

    def predict(
        self,
        backend: TextModel,
        mode: str,
        rows: list[dict],
        new_text: str,
        orchestrator_note: str = "",
        row_model: str | None = None,
        orchestrator_model: str | None = None,
        progress=None,
    ) -> str:
        if mode not in {"writing", "reviewing"}:
            raise ValueError("mode must be writing or reviewing")
        if not new_text.strip():
            raise ValueError("New document text is empty")
        usable = [row for row in rows if _row_is_complete(row)][:MAX_ROWS]
        findings: list[str] = []
        total = len(usable)
        for index, row in enumerate(usable, start=1):
            step = f"row-{index}"
            title = f"History row {index} of {total}"
            _emit(
                progress,
                kind="start",
                step=step,
                title=title,
                model=row_model or "",
            )
            prompt = _row_prompt(self.row_brief, mode, row)
            finding = backend.complete(
                prompt,
                row_model,
                on_delta=_listener(progress, step),
            ).strip()
            findings.append(finding)
            _emit(progress, kind="done", step=step, reply=finding)
        label = "feedback" if mode == "reviewing" else "protocol"
        _emit(
            progress,
            kind="start",
            step="orchestrator",
            title=f"Orchestrator, writing the {label}",
            model=orchestrator_model or "",
            received=_received(findings),
        )
        master = (
            self.master_writing if mode == "writing" else self.master_reviewing
        )
        prompt = _master_prompt(
            master,
            findings,
            new_text,
            orchestrator_note,
        )
        body = backend.complete(
            prompt,
            orchestrator_model,
            on_delta=_listener(progress, "orchestrator"),
        ).strip()
        _emit(progress, kind="done", step="orchestrator", reply=body)
        if not body:
            raise RuntimeError("Model returned an empty document")
        return body


def _emit(progress, **payload) -> None:
    if progress is not None:
        progress(payload)


def _listener(progress, step: str):
    if progress is None:
        return None

    def on_delta(channel: str, text: str) -> None:
        if channel in {"thinking", "reply"} and text:
            _emit(
                progress,
                kind="delta",
                step=step,
                channel=channel,
                text=text,
            )

    return on_delta


def _received(findings: list[str]) -> str:
    if not findings:
        return "No history rows. This call only sees the new document."
    blocks = [
        f"Row {index}:\n{text}" for index, text in enumerate(findings, 1)
    ]
    return "\n\n".join(blocks)


def _row_is_complete(row: dict) -> bool:
    left = str(row.get("left") or "").strip()
    middle = str(row.get("middle") or "").strip()
    return bool(left and middle)


_ROW_LABELS = {
    "writing": ("Raw notes", "Finished protocol", "Review feedback"),
    "reviewing": ("First draft", "Final protocol", "Feedback email"),
}


def _row_prompt(brief: str, mode: str, row: dict) -> str:
    note = str(row.get("note") or "").strip()
    left, middle, right = _ROW_LABELS[mode]
    parts = [
        brief,
        f"Mode: {mode}",
        f"{left}:\n" + str(row.get("left") or "").strip(),
        f"{middle}:\n" + str(row.get("middle") or "").strip(),
        f"{right}:\n" + str(row.get("right") or "").strip(),
    ]
    if note:
        parts.append("Additional context:\n" + note)
    return "\n\n".join(parts)


def _master_prompt(
    brief: str,
    findings: list[str],
    new_text: str,
    orchestrator_note: str,
) -> str:
    blocks = [brief, "Conclusions from earlier rows:"]
    if findings:
        for index, finding in enumerate(findings, start=1):
            blocks.append(f"Row {index}:\n{finding}")
    else:
        blocks.append("No earlier rows were provided.")
    blocks.append("New document:\n" + new_text.strip())
    note = orchestrator_note.strip()
    if note:
        blocks.append("Additional context:\n" + note)
    return "\n\n".join(blocks)
