# logger.py
from torch.utils.tensorboard import SummaryWriter
import datetime
import os

class Logger:
    def __init__(self, project_name, experiment_name=None, log_dir="logs"):
        """
        Initializes the TensorBoard logger.

        Args:
            project_name (str): Name of the project.
            experiment_name (str, optional): Specific experiment name.
            log_dir (str): Directory to save logs.
        """
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.log_path = os.path.join(log_dir, project_name, experiment_name or timestamp)
        self.writer = SummaryWriter(log_dir=self.log_path)

    def log_embedding(self, latent, metadata, global_step, tag, header):
        self.writer.add_embedding(latent, metadata=metadata.tolist(),global_step=global_step, tag=tag, metadata_header = header)

    def define_custom_layout(self, layout):
        self.writer.add_custom_scalars(layout)

    def log_scalar(self, tag, value, step):
        """Logs a scalar value (e.g., loss, accuracy)."""
        self.writer.add_scalar(tag, value, step)

    def log_text(self, tag, text, step):
        """Logs text data."""
        self.writer.add_text(tag, text, step)

    def log_image(self, tag, image, step):
        """Logs an image (e.g., input samples or predictions)."""
        self.writer.add_image(tag, image, step)

    def log_figure(self, tag, image, step):
        """Logs an image (e.g., input samples or predictions)."""
        self.writer.add_figure(tag, image, step)


    def log_histogram(self, tag, values, step):
        """Logs a histogram (e.g., model weights or gradients)."""
        self.writer.add_histogram(tag, values, step)

    def log_graph(self, model, inputs):
        """Logs the model graph."""
        self.writer.add_graph(model, inputs)

    def flush(self):
        """Flushes the logged data."""
        self.writer.flush()

    def close(self):
        """Closes the writer."""
        self.writer.close()

    def get_log_path(self):
        """Returns the path to the logs."""
        return self.log_path
