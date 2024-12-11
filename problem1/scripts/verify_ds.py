import os
import click



@click.command()
@click.option('--source', '-s', required=True, type=click.Path(exists=True), help='Path to the dataset')
def main(source):
    dirs_to_crawl = [
        "boring",
        "digraph",
        "entropy",
        "spiral",
        "original"
    ]
    for _dir in dirs_to_crawl:
        full_dir = os.path.join(source, _dir)
        # get all files in the directory, recursively
        all_files = []
        for root, dirs, files in os.walk(full_dir):
            for file in files:
                all_files.append(os.path.join(root, file))
        # there should 5_500 files in each directory
        assert len(all_files) == 5500, f"Expected 5500 files in {full_dir}, got {len(all_files)}"


if __name__ == "__main__":
    main()
