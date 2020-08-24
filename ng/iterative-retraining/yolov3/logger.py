"""
Simple implementation of a Tensorboard SummaryWriter
"""
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Custom logger wrapper for scalars."""

    def __init__(self, log_dir, title):
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{title}")

    def scalar_summary(self, tag, value, step):
        """Add a value to a timeseries metric."""
        self.writer.add_scalar(tag, value, step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Add a list of tag-value pairs from a given step to the summary."""
        for tag, value in tag_value_pairs:
            self.writer.add_scalar(tag, value, step)
