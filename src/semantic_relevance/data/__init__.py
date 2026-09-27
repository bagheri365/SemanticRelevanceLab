"""Dataset loading and domain models."""

from .models import Document, Query
from .trec_covid import TrecCovidDataset, load_trec_covid

__all__ = ["Document", "Query", "TrecCovidDataset", "load_trec_covid"]
