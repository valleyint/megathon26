"""Unit tests for evaluation metrics."""

import unittest
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    average_precision,
    compute_metrics,
)


class TestMetrics(unittest.TestCase):
    def test_precision_recall_at_k(self):
        predicted = ["cat1", "cat2", "cat3", "cat4"]
        ground_truth = {"cat1", "cat3", "cat5"}

        self.assertEqual(precision_at_k(predicted, ground_truth, k=1), 1.0)
        self.assertEqual(precision_at_k(predicted, ground_truth, k=2), 0.5)
        self.assertAlmostEqual(precision_at_k(predicted, ground_truth, k=4), 2 / 4.0)

        self.assertAlmostEqual(recall_at_k(predicted, ground_truth, k=1), 1 / 3.0)
        self.assertAlmostEqual(recall_at_k(predicted, ground_truth, k=3), 2 / 3.0)

    def test_average_precision(self):
        predicted = ["cat1", "cat2", "cat3"]
        ground_truth = {"cat1", "cat3"}
        # Hit at 1 (prec 1/1), Miss at 2, Hit at 3 (prec 2/3)
        # AP = (1.0 + 2/3) / 2 = 5/6 = 0.8333...
        ap = average_precision(predicted, ground_truth)
        self.assertAlmostEqual(ap, 5 / 6.0)

    def test_compute_metrics_batch(self):
        preds = [["cat1", "cat2"], ["cat3", "cat4"]]
        gts = [{"cat1"}, {"cat4"}]
        metrics = compute_metrics(preds, gts, k_values=(1, 2), latencies_ms=[10.0, 20.0])

        self.assertEqual(metrics.num_queries, 2)
        self.assertEqual(metrics.precision_at_k[1], 0.5)
        self.assertEqual(metrics.precision_at_k[2], 0.5)
        self.assertEqual(metrics.recall_at_k[1], 0.5)
        self.assertEqual(metrics.recall_at_k[2], 1.0)
        self.assertEqual(metrics.mean_latency_ms, 15.0)


if __name__ == "__main__":
    unittest.main()
