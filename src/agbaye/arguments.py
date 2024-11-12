import os
from dataclasses import dataclass, field
from typing import Literal, Optional

import datatrove.executor


@dataclass
class MainArgs:
    dump_name: str
    output_path: str
    skip_warc_rows: int = 0
    logging_dir: Optional[str] = None
    executor_class: Literal["local", "slurm"] = "local"
    randomize_start_duration: int = 180

    def __post_init__(self):
        self.executor_class = getattr(datatrove.executor, f"{self.executor_class.capitalize()}PipelineExecutor")


@dataclass
class ADLFSArgs:
    use_adlfs: bool = False
    account_name: Optional[str] = None
    account_key: Optional[str] = None

    def __post_init__(self):
        if self.use_adlfs:
            from dotenv import load_dotenv

            load_dotenv()

            assert self.account_name is not None, "account_name is required when use_adlfs is True"

            self.account_key = self.account_key or os.getenv("AZURE_STORAGE_KEY")
            assert self.account_key is not None, "account_key is required when use_adlfs is True"


@dataclass
class SlurmArgs:
    job_name: Optional[str] = None
    tasks: int = 1
    slurm_logs_folder: str = "slurm_logs"
    time: str = "24:00:00"
    partition: Optional[str] = None
    mail_user: Optional[str] = None
    cpus_per_task: int = 1,
    mem_per_cpu_gb: int = 2
    sbatch_args: Optional[dict | str] = None
    srun_args: Optional[dict | str] = None

    def __post_init__(self):
        if isinstance(self.sbatch_args, str):
            ...
        if isinstance(self.srun_args, str):
            ...


@dataclass
class LIDArgs:
    device: str = "cpu"
    threshold: float = 0.65
    lid_backend: str = "afrolid"
    lid_batch_size: int = 1
    lid_keep_top_predictions_threshold: float = -1
