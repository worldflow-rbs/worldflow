import unittest

from worldflow.training.curriculum import TrainingBatchSampler, domain_probabilities


class IndexedExamples:
    source_indices = {
        "sim7": list(range(12)),
        "molmo_droid": list(range(12, 24)),
        "molmo_ytvis": list(range(24, 36)),
        "demo5": list(range(36, 48)),
        "agibot": list(range(48, 60)),
        "epic": list(range(60, 72)),
    }

    def __len__(self):
        return sum(map(len, self.source_indices.values()))


class CurriculumTests(unittest.TestCase):
    def test_reference_transitions(self):
        self.assertEqual(domain_probabilities(0, 51100)["sim"], 0.8)
        self.assertAlmostEqual(domain_probabilities(12000, 51100)["sim"], 0.45)
        self.assertEqual(domain_probabilities(24050, 51100)["outdoor"], 0.5)
        self.assertEqual(domain_probabilities(51100, 51100)["demo5"], 0.5)

    def test_resume_after_prefetch(self):
        def batches(rank, start=0):
            return list(
                TrainingBatchSampler(
                    IndexedExamples(),
                    stage=2,
                    batch_size=3,
                    total_steps=8,
                    grad_accum=2,
                    rank=rank,
                    world_size=2,
                    seed=42,
                    start_step=start,
                )
            )

        self.assertEqual(batches(0)[6:], batches(0, 3))
        self.assertNotEqual(batches(1), batches(0))

    def test_partial_domain_manifest_requires_explicit_uniform_sampling(self):
        dataset = IndexedExamples()
        dataset.source_indices = {"demo5": [0]}
        args = dict(stage=2, batch_size=1, total_steps=1, grad_accum=1, rank=0, world_size=1, seed=42)
        with self.assertRaisesRegex(ValueError, "requires all five domains"):
            TrainingBatchSampler(dataset, **args)
        self.assertEqual(list(TrainingBatchSampler(dataset, **args, sampling="uniform")), [[0]])

    def test_warmup_shuffles_without_replacement_and_resumes_inside_an_epoch(self):
        dataset = IndexedExamples()
        dataset.source_indices = {"demo5": list(range(7))}
        args = dict(stage=1, batch_size=2, total_steps=6, grad_accum=1, rank=0, world_size=1, seed=10)
        batches = list(TrainingBatchSampler(dataset, **args))
        self.assertEqual(len(set(sum(batches[:3], []))), 6)
        self.assertEqual(batches[2:], list(TrainingBatchSampler(dataset, **args, start_step=2)))
