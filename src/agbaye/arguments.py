import os

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class MainArgs:
    dump_name: str
    output_path: str
    executor_class: Literal["local", "slurm"] = "local"


@dataclass
class ADLFSArgs:
    use_adlfs: bool = False
    account_name: Optional[str] = None
    account_key: Optional[str] = None

    def __post_init__(self):
        if self.use_adlfs:
            assert self.account_name is not None, "account_name is required when use_adlfs is True"

            self.account_key = self.account_key or os.getenv("AZURE_STORAGE_KEY")
            assert self.account_key is not None, "account_key is required when use_adlfs is True"


@dataclass
class SlurmArgs:
    tasks: int
    slurm_logs_folder: str
    time: str = "24:00:00"
    partition: Optional[str] = None
    mail_user: Optional[str] = None
    cpus_per_task: int = 1,
    mem_per_cpu_gb: int = 2
    randomize_start_durations: int = 180
    sbatch_args: Optional[dict | str] = None
    srun_args: Optional[dict | str] = None

    def __post_init__(self):
        if isinstance(self.sbatch_args, str):
            ...
        if isinstance(self.srun_args, str):
            ...


@dataclass
class LIDArgs:
    lid_backend: str = "afrolid"
    lid_batch_size: int = 1
    lid_keep_top_pairs_threshold: float = -1
