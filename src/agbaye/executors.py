from dataclasses import dataclass, asdict
from typing import Optional

from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.io import DataFolderLike
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    GopherQualityFilter,
    GopherRepetitionFilter,
    URLFilter
)
from datatrove.pipeline.readers import WarcReader
from datatrove.pipeline.writers import JsonlWriter

from .lid import AfricanLanguageFilter


@dataclass
class SlurmArgs:
    tasks: int
    slurm_logs_folder: str
    time: str = "24:00:00"
    partition: Optional[str] = None
    mail_user: Optional[str] = None
    mem_per_cpu_gb: int = 2
    randomize_start_durations: int = 180
    sbatch_args: Optional[dict] = None
    srun_args: Optional[dict] = None


def get_common_crawl_executor(
    dump_name: str,
    output_path: DataFolderLike,
    slurm_args: SlurmArgs,
    language_threshold: float = 0.65,
    lid_backend: str = "afrolid",
    lid_batch_size: int = 1,
    lid_keep_top_pairs_threshold: float = -1,
) -> None:
    executor = SlurmPipelineExecutor(
        job_name=f"cc_{dump_name}",
        pipeline=[
            WarcReader(
                f"s3://commoncrawl/crawl-data/{dump_name}/segments",
                glob_pattern="*/warc/*",
                default_metadata={"dump_name": dump_name},
            ),
            URLFilter(exclusion_writer=JsonlWriter(f"{output_path}/removed/url/{dump_name}")),
            Trafilatura(favour_precision=True),
            AfricanLanguageFilter(
                backend=lid_backend,
                batch_size=lid_batch_size,
                language_threshold=language_threshold,
                keep_top_pairs_threshold=lid_keep_top_pairs_threshold
            ),
            GopherRepetitionFilter(exclusion_writer=JsonlWriter(f"{output_path}/removed/repetitive/{dump_name}")),
            GopherQualityFilter(exclusion_writer=JsonlWriter(f"{output_path}/removed/quality/{dump_name}")),
            JsonlWriter(f"{output_path}/output/{dump_name}")
        ]
        **asdict(slurm_args)
    )

    executor.run()
