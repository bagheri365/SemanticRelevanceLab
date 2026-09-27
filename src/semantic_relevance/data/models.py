"""Domain models for information-retrieval datasets."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Document:
    """A searchable document."""

    document_id: str
    title: str
    text: str

    @property
    def searchable_text(self) -> str:
        """Combine title and body for simple retrieval baselines."""
        return " ".join(part for part in (self.title, self.text) if part).strip()


@dataclass(frozen=True, slots=True)
class Query:
    """A search query."""

    query_id: str
    text: str
