import os
from typing import Protocol, BinaryIO, Optional


class CheckpointStorage(Protocol):
    """
    CheckpointStorage is the interface for CheckpointStorage implementations.
    """

    def save_checkpoint(self, buffer: BinaryIO, key: str) -> None: ...


def init_storage(storage: str, base_path: Optional[str] = None) -> CheckpointStorage:
    """Initialize a checkpoint storage backend.

    Args:
        storage: Storage type - "local" or "azure"
        base_path: Base path for local storage (required for "local", ignored for "azure")

    Returns:
        CheckpointStorage implementation
    """
    match storage:
        case "azure":
            return AzureCheckpointStorage()
        case "local":
            return LocalCheckpointStorage(base_path=base_path or ".")
        case _:
            raise ValueError(f"Unknown storage type: {storage}")


class AzureCheckpointStorage:
    """
    AzureCheckpointStorage saves checkpoints to Azure Blob Storage.
    """

    def save_checkpoint(self, buffer: BinaryIO, key: str) -> None:
        # TODO: Implement Azure Blob Storage upload
        raise NotImplementedError("Azure checkpoint storage not yet implemented")


class LocalCheckpointStorage:
    """
    LocalCheckpointStorage saves checkpoints to the local filesystem.
    """

    def __init__(self, base_path: str = "."):
        """Initialize local checkpoint storage.

        Args:
            base_path: Base directory for storing checkpoints
        """
        self.base_path = base_path

    def save_checkpoint(self, buffer: BinaryIO, key: str) -> None:
        """Save a checkpoint to the local filesystem.

        Args:
            buffer: Binary buffer containing the serialized checkpoint
            key: Relative path/key for the checkpoint file
        """
        path = os.path.join(self.base_path, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(buffer.read())
