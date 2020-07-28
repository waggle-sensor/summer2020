from tensorboardX import SummaryWriter


class Logger(object):
    def __init__(self, log_dir, title):
        self.writer = SummaryWriter(log_dir, comment=title)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        for tag, value in tag_value_pairs:
            self.writer.add_scalar(tag, value, step)
