from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MainArgs:
    dump_name: str
    output_path: str


@dataclass
class ADLFSArgs:
    use_adlfs: bool = False
    account_name: Optional[str] = None
    account_key: Optional[str] = None


@dataclass
class SlurmArgs:
    tasks: int
    slurm_logs_folder: str
    time: str = "24:00:00"
    partition: Optional[str] = None
    mail_user: Optional[str] = None
    mem_per_cpu_gb: int = 2
    randomize_start_durations: int = 180
    sbatch_args: Optional[dict | str] = None
    srun_args: Optional[dict | str] = None


@dataclass
class LIDArgs:
    lid_backend: str = "afrolid"
    lid_batch_size: int = 1
    lid_keep_top_pairs_threshold: float = -1
