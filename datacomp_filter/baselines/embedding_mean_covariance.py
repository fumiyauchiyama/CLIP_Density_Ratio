"""
This is a command line script for clustering image embeddings for the DataComp pool.
The output of the script is a numpy file containing the computed cluster centers.
Please see image_based_clustering.md for additional information, and note that we also provide precomputed numpy files with the cluster centers used in the DataComp baselines.
"""

import argparse
import multiprocessing as mp
from functools import partial
from multiprocessing import Pool
from typing import Any, List, Tuple

import fasttext
import fsspec
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from baselines.apply_filter import caption_filter
from baselines.utils import download, random_seed

torch.backends.cudnn.benchmark = True


def calc_statistics(
    embeddings: np.ndarray,
) -> torch.Tensor:
    """Returns the mean and covariance of the embeddings

    Args:
        embeddings (np.ndarray): embeddings to cluster
    """

    return np.mean(embeddings, axis=0), np.cov(embeddings, rowvar=False)


def load_embedding_helper(
    fs_root: Tuple[Any, str],
    key: str = "l14_img",
    caption_filtering: bool = False,
    sample_ratio: float = -1.0,
) -> np.ndarray:
    """worker function to load embeddings

    Args:
        fs_root (Tuple[Any, str]): (filesystem, path_root)
        key (str, optional): key to load from npz. Defaults to "l14_img".
        caption_filtering (bool, optional): whether to enable caption filter. Defaults to False.
        sample_ratio (float, optional): ratio of samples to use. Defaults to -1.0.
    """

    fs, path_root = fs_root
    embed = np.load(fs.open(f"{path_root}.npz"))[key]
    if caption_filtering:
        lang_detect_model = fasttext.load_model(
            download("fasttext", "~/.cache/fasttext")
        )
        df = pd.read_parquet(
            f"{path_root}.parquet", columns=["uid", "text"], filesystem=fs
        )
        mask = caption_filter(df, lang_detect_model)
        embed = embed[mask]
    if sample_ratio > 0:
        n = len(embed)
        idx = np.random.choice(range(n), size=int(n * sample_ratio))
        embed = embed[idx]
    return embed


def load_embedding(
    paths: List[Tuple[Any, str]],
    n_workers: int = 10,
    key: str = "l14_img",
    caption_filtering: bool = False,
    sample_ratio: float = -1.0,
) -> np.ndarray:
    """worker function to load embeddings

    Args:
        paths (List[Tuple[Any, str]]): list of (filesystem, path_root)
        n_workers (int, optional): number of workers. Defaults to 10.
        key (str, optional): key to load from npz. Defaults to "l14_img".
        caption_filtering (bool, optional): whether to enable caption filter. Defaults to False.
        sample_ratio (float, optional): ratio of samples to use. Defaults to -1.0.
    """
    mp.set_start_method("spawn", force=True)
    print("start loading embedding")
    worker = partial(
        load_embedding_helper,
        key=key,
        caption_filtering=caption_filtering,
        sample_ratio=sample_ratio,
    )

    with Pool(n_workers) as pool:
        embeds = [
            res
            for res in tqdm(
                pool.imap(worker, paths), total=len(paths)
            )  # imap so that it can be reproduced
            if len(res) > 0
        ]
    return np.vstack(embeds)


def get_mean_cov(
    paths: List[Tuple[Any, str]],
    embedding_key: str = "l14_img",
    num_workers: int = 10,
    caption_filtering: bool = False,
    sample_ratio: float = -1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    embeddings = load_embedding(
        paths,
        key=embedding_key,
        n_workers=num_workers,
        caption_filtering=caption_filtering,
        sample_ratio=sample_ratio,
    )

    embeddings = embeddings.astype(np.float32)
    mean, cov = calc_statistics(embeddings)
    print(f"{embedding_key}'s mean: {mean.shape} | cov: {cov.shape}")
    return mean, cov



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_dir",
        type=str,
        help="directory (local or cloud) containing parquet, npz metadata",
    )
    parser.add_argument("--save_path", type=str, help="local path to output centroids")
    parser.add_argument(
        "--arch",
        default="l14",
        type=str,
        choices=["l14", "b32"],
        help="precomputed embeddings used for clustering",
    )
    parser.add_argument(
        "--sample_ratio",
        default=-1.0,
        type=float,
        help="ratio of samples to use (we need to sample because of memory constraint)",
    )
    parser.add_argument(
        "--num_workers",
        default=10,
        type=int,
        help="number of workers, generally set to number of cpu cores",
    )
    parser.add_argument(
        "--disable_caption_filtering",
        default=False,
        action="store_true",
        help="whether to disable text-based basic filtering",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
    )
    args = parser.parse_args()

    random_seed(args.seed)

    sample_ratio = args.sample_ratio
    caption_filtering = not args.disable_caption_filtering

    fs, url = fsspec.core.url_to_fs(args.metadata_dir)
    paths = [(fs, str(x.split(".parquet")[0])) for x in fs.ls(url) if ".parquet" in x]

    print(f"caption filtering: {caption_filtering} | sample_ratio={sample_ratio}")

    img_mean, img_cov = get_mean_cov(
        paths,
        embedding_key=f"{args.arch}_img",
        num_workers=args.num_workers,
        caption_filtering=caption_filtering,
        sample_ratio=sample_ratio,
    )

    txt_mean, txt_cov = get_mean_cov(
        paths,
        embedding_key=f"{args.arch}_txt",
        num_workers=args.num_workers,
        caption_filtering=caption_filtering,
        sample_ratio=sample_ratio,
    )

    # save the mean and covariance to a numpy file
    np.savez_compressed(
        args.save_path,
        img_mean=img_mean,
        img_cov=img_cov,
        txt_mean=txt_mean,
        txt_cov=txt_cov,
    )
    print(f"saved to {args.save_path}")


