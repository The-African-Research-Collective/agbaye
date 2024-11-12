import os
import sys

from transformers import HfArgumentParser

from .arguments import ADLFSArgs, LIDArgs, MainArgs, SlurmArgs
from .executors import get_common_crawl_executor


def main():
    parser = HfArgumentParser((MainArgs, ADLFSArgs, LIDArgs, SlurmArgs))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        main_args, adlfs_args, lid_args, slurm_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        main_args, adlfs_args, lid_args, slurm_args = parser.parse_args_into_dataclasses()

    if main_args.executor_class.__name__ == "SlurmPipelineExecutor":
        assert slurm_args.job_name is not None, "job_name is required when executor_class is SlurmPipelineExecutor"

    fs = None
    if adlfs_args.use_adlfs:
        from adlfs import AzureBlobFileSystem

        fs = AzureBlobFileSystem(account_name=adlfs_args.account_name)

    executor = get_common_crawl_executor(
        executor_class=main_args.executor_class,
        dump_name=main_args.dump_name,
        logging_dir=main_args.logging_dir,
        skip_warc_rows=main_args.skip_warc_rows,
        randomize_start_duration=main_args.randomize_start_duration,
        output_path=main_args.output_path,
        language_threshold=lid_args.threshold,
        lid_backend=lid_args.lid_backend,
        lid_batch_size=lid_args.lid_batch_size,
        lid_device=lid_args.device,
        lid_keep_top_predictions_threshold=lid_args.lid_keep_top_predictions_threshold,
        fs=fs,
        **(vars(slurm_args) if main_args.executor_class == "slurm" else {})
    )

    executor.run()


if __name__ == "__main__":
    main()
