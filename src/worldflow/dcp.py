"""Read trusted local PyTorch DCP checkpoints across supported Python versions."""

import pickle
from pathlib import Path

from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import StorageMeta


class _MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Python 3.13 moved the pathlib implementation. Older interpreters
        # expose these same public classes directly under the pathlib module.
        if module == "pathlib._local" and name in (
            "Path",
            "PosixPath",
            "WindowsPath",
            "PurePath",
            "PurePosixPath",
            "PureWindowsPath",
        ):
            module = "pathlib"
        return super().find_class(module, name)


class _LocalDCPReader(FileSystemReader):
    def read_metadata(self):
        try:
            return super().read_metadata()
        except ModuleNotFoundError as error:
            if error.name != "pathlib._local":
                raise
        metadata_path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(metadata_path, "rb") as stream:
            metadata = _MetadataUnpickler(stream).load()
        # Preserve the load identity supplied by PyTorch's FileSystemReader.
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        return metadata


def dcp_reader(path: Path) -> FileSystemReader:
    """Create a filesystem reader for a trusted training or backbone checkpoint."""
    return _LocalDCPReader(path)
