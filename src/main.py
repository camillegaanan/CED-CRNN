import argparse
import os
from data.reader import Dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)

    parser.add_argument("--transform", action="store_true", default=False)
    args = parser.parse_args()

    raw_path = os.path.join("..", "raw", args.source)
    source_path = os.path.join("..", "data", f"{args.source}.hdf5")

    input_size = (1024, 128, 1)
    max_text_length = 128

    if args.transform:
        print(f"{args.source} dataset will be transformed...")
        ds = Dataset(source=raw_path, name=args.source)
        ds.read_partitions()
        ds.save_partitions(source_path, input_size, max_text_length)
