"""Runtime dependencies for the document agent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import UUID

from app.retrieval.retriever import DocumentRetriever
from app.retrieval.types import RetrievedPassage

StatusCallback = Callable[[str, str], None]


@dataclass
class TurnRegistry:
    """Tracks every chunk retrieved during a turn — the citation allowlist."""

    passages_by_chunk_id: dict[UUID, RetrievedPassage] = field(default_factory=dict)

    def register(self, passage: RetrievedPassage) -> None:
        self.passages_by_chunk_id[passage.chunk_id] = passage
        for neighbor in passage.neighbors:
            self.passages_by_chunk_id[neighbor.chunk_id] = neighbor

    def register_many(self, passages: list[RetrievedPassage]) -> None:
        for passage in passages:
            self.register(passage)


@dataclass
class DocumentAgentDeps:
    retriever: DocumentRetriever
    registry: TurnRegistry
    thread_id: UUID
    user_id: UUID
    #: Which retrieval mode this turn runs in; see retrieval/modes.py.
    retrieval_mode: str | None = None
    on_status: StatusCallback | None = None
    #: Agentic mode only. The agent's data storage: a key the model chose, to
    #: the full text of a page it parsed. Documents live here rather than in the
    #: context window, and are read by prompting a sub-model about them.
    fetched_documents: dict[str, str] = field(default_factory=dict)
    #: Agentic mode only. Document key to the URL it came from, so a cited
    #: passage can name its source.
    document_urls: dict[str, str] = field(default_factory=dict)
    #: Concatenated tool returns for the turn, filled in after the agent runs.
    #: The grounding gate uses it to check figures in an answer that cites
    #: nothing, which was previously trusted without evidence.
    #: Whether the model may think before answering, chosen per question in the
    #: UI. None means "leave the model on its configured default". Off is the
    #: default everywhere: measured on the local model, thinking multiplied
    #: output tokens 30x and produced an identical tool call.
    thinking: bool | None = None
    #: Which registry model is answering this turn. Sub-model calls follow it,
    #: so a reasoning step runs on the model the user actually chose rather than
    #: on whatever endpoint happens to be configured.
    model_name: str = ""
    #: CAG only. Which company's filing is preloaded into the instructions,
    #: chosen before the turn starts. It cannot be a tool argument: a tool result
    #: arrives AFTER the question, and CAG's whole economic argument is that the
    #: filing is a stable prefix the KV cache reuses across questions.
    cag_ticker: str = ""
    cag_fiscal_year: int = 0
    tool_outputs: str = ""
    #: Agentic mode only. `extract_from_filing` reads filing slices with its own
    #: sub-model call, which never appears in the run's usage, so without this
    #: the mode looks far cheaper than it is. Counted here and reported with it.
    extract_calls: int = 0
    extract_input_tokens: int = 0
    extract_output_tokens: int = 0

    def emit_status(self, stage: str, message: str) -> None:
        if self.on_status is not None:
            self.on_status(stage, message)
