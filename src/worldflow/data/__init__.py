"""Training input adapters for WorldFlow."""

from .dataset import TrainingDataset, collate_training_samples

__all__ = ["TrainingDataset", "collate_training_samples"]
