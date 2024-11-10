from typing import Any

from datatrove.executor.base import PipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor
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


def get_common_crawl_executor(
    dump_name: str,
    output_path: DataFolderLike,
    language_threshold: float = 0.65,
    lid_backend: str = "afrolid",
    lid_batch_size: int = 1,
    lid_keep_top_pairs_threshold: float = -1,
    executor_class: PipelineExecutor = LocalPipelineExecutor,
    **kwargs: Any
) -> PipelineExecutor:
    executor = executor_class(
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
            # TODO: @theyorubayesian - GopherQualityFilter requires language specific information for tokenization
            # GopherQualityFilter(exclusion_writer=JsonlWriter(f"{output_path}/removed/quality/{dump_name}")),
            JsonlWriter(f"{output_path}/output/{dump_name}")
        ]
        **kwargs
    )

    return executor
