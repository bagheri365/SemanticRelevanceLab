import json
import random

from semantic_relevance.evaluation import (
    ExperimentResult,
    runtime_metadata,
    set_seed,
)


def test_experiment_result_saves_json(tmp_path) -> None:
    result = ExperimentResult(
        experiment="toy_run",
        metrics={"ndcg@10": 0.75},
        latency={"p95_ms": 12.5},
        metadata={"seed": 42},
    )

    output = result.save_json(tmp_path / "nested" / "result.json")
    payload = json.loads(output.read_text())

    assert payload["experiment"] == "toy_run"
    assert payload["metrics"]["ndcg@10"] == 0.75
    assert payload["latency"]["p95_ms"] == 12.5
    assert payload["metadata"]["seed"] == 42


def test_runtime_metadata_contains_reproducibility_fields() -> None:
    metadata = runtime_metadata()
    assert metadata["python"]
    assert metadata["python_implementation"]
    assert metadata["platform"]
    assert metadata["executable"]


def test_set_seed_makes_random_repeatable() -> None:
    set_seed(7)
    first = [random.random() for _ in range(3)]
    set_seed(7)
    second = [random.random() for _ in range(3)]
    assert first == second
