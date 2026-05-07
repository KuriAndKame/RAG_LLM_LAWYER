# evaluation/metrics.py
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    AnswerCorrectness,
)


def get_ragas_metrics():
    return [
        ContextPrecision(),
        ContextRecall(),
        AnswerRelevancy(),
        Faithfulness(),
        AnswerCorrectness(),
    ]
