from transformers import HfArgumentParser

from .arguments import ADLFSArgs, LIDArgs, SlurmArgs
from .executors import get_common_crawl_executor


def main():
    parser = HfArgumentParser((ADLFSArgs, LIDArgs, SlurmArgs))
    main_args, adlfs_args, lid_args, slurm_args = parser.parse_args_into_dataclasses()

    output_path = main_args.output_path
    if adlfs_args.use_adlfs:
        from adlfs import AzureBlobFileSystem
        from datatrove.io import DataFolder

        fs = AzureBlobFileSystem(account_name=adlfs_args.account_name)
        output_path = DataFolder(main_args.output_path, fs)

    executor = get_common_crawl_executor(
        dump_name=main_args.dump_name,
        output_path=output_path,
        lid_backend=lid_args.lid_backend,
        lid_batch_size=lid_args.lid_batch_size,
        lid_keep_top_pairs_threshold=lid_args.lid_keep_top_pairs_threshold,
        **vars(slurm_args)
    )

    executor.run()