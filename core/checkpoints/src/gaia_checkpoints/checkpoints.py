import threading
import queue
import io
import torch
import logging
from typing import Optional
from .storage import CheckpointStorage, init_storage


class CheckpointSaver:
    """Async checkpoint saver that writes checkpoints in a background thread."""

    def __init__(
        self,
        prefix: str,
        storage: str = "local",
        base_path: Optional[str] = None,
        max_queue_size: int = 3,
    ):
        """Initialize the checkpoint saver.

        Args:
            prefix: Prefix path for checkpoint files
            storage: Storage backend type - "local" or "azure"
            base_path: Base path for local storage (required for "local", ignored for "azure")
            max_queue_size: Maximum number of checkpoints to queue (older ones skipped if full)
        """
        self.storage = init_storage(storage, base_path=base_path)
        self.prefix = prefix
        self.queue = queue.Queue(maxsize=max_queue_size)

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def save(
        self,
        global_step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        filename: str,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ) -> None:
        """Queue a checkpoint for saving.

        Args:
            global_step: Current training step
            model: Model to save
            optimizer: Optimizer to save
            filename: Filename for the checkpoint
            scheduler: Optional learning rate scheduler to save
        """
        checkpoint = {
            "step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        if scheduler is not None:
            checkpoint["scheduler"] = scheduler.state_dict()

        try:
            self.queue.put_nowait((checkpoint, filename))
        except queue.Full:
            logging.warning("Checkpoint queue full; skipping checkpoint")

    def _worker(self):
        """Background worker that processes the checkpoint queue."""
        while True:
            checkpoint, filename = self.queue.get()
            try:
                buffer = io.BytesIO()
                torch.save(checkpoint, buffer)
                buffer.seek(0)

                key = f"{self.prefix}/{filename}"
                self.storage.save_checkpoint(buffer, key)
            except Exception:
                logging.exception("Checkpoint upload failed")
            finally:
                self.queue.task_done()
