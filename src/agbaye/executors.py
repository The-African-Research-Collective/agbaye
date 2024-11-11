from typing import Any

import torch
from datatrove.executor.base import PipelineExecutor
from datatrove.executor import LocalPipelineExecutor
from datatrove.io import get_datafolder, DataFolder
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    GopherQualityFilter,
    GopherRepetitionFilter,
    URLFilter
)
from datatrove.pipeline.readers import WarcReader
from datatrove.pipeline.writers import JsonlWriter
from fsspec import AbstractFileSystem

from .lid import AfricanLanguageFilter


def _get_datafolder(output_path: str, fs: AbstractFileSystem | None = None) ->  DataFolder:
    if fs is None:
        return get_datafolder(output_path)
    return get_datafolder((output_path, fs))


def get_common_crawl_executor(
    dump_name: str,
    output_path: str,
    language_threshold: float = 0.65,
    lid_backend: str = "afrolid",
    lid_batch_size: int = 1,
    lid_keep_top_predictions_threshold: float = -1,
    lid_device: str | torch.device = "cpu",
    executor_class: PipelineExecutor = LocalPipelineExecutor,
    fs: AbstractFileSystem | None = None,
    **kwargs: Any
) -> PipelineExecutor:
    executor = executor_class(
        pipeline=[
            WarcReader(
                f"s3://commoncrawl/crawl-data/{dump_name}/segments",
                glob_pattern="*/warc/*",
                default_metadata={"dump_name": dump_name},
            ),
            URLFilter(),
            Trafilatura(favour_precision=True, timeout=1.0),
            AfricanLanguageFilter(
                backend=lid_backend,
                batch_size=lid_batch_size,
                language_threshold=language_threshold,
                keep_top_predictions_threshold=lid_keep_top_predictions_threshold,
                device=lid_device
            ),

            # TODO: @theyorubayesian - Gopher Filters requires language specific information for tokenization
            # GopherRepetitionFilter(exclusion_writer=JsonlWriter(f"{output_path}/removed/repetitive/{dump_name}")),
            # GopherQualityFilter(exclusion_writer=JsonlWriter(f"{output_path}/removed/quality/{dump_name}")),

            JsonlWriter(_get_datafolder(f"{output_path}/output/{dump_name}", fs))
        ],
        **kwargs
    )

    return executor
