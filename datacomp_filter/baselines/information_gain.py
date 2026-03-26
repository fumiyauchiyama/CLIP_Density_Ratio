import torch
import numpy as np
import pandas as pd
import multiprocessing as mp
from queue import Empty
import os
import time
import fsspec
from tqdm import tqdm
import argparse
from typing import Union, Final
from enum import Enum
import math
from torch.nn import functional as F

class Mode(Enum):
    CENTERIZED = "CENTERIZED"
    WHITEN = "WHITEN"
    MC = "MC"
    REVERSE_MC = "REVERSE_MC"

def is_2504(mode: Mode) -> bool:
    return mode in (Mode.CENTERIZED, Mode.WHITEN)

def is_2509(mode: Mode) -> bool:
    return mode in (Mode.MC, Mode.REVERSE_MC)

b32_logit_scale: Final[float] = 4.6052
b32_logit_bias: Final[float] = 0.0
l14_logit_scale: Final[float] = 4.6052
l14_logit_bias: Final[float] = 0.0

@torch.no_grad()
def get_information_gain_2504(
    features: torch.Tensor, 
    mean: torch.Tensor, 
    cov_another: torch.Tensor,
    mode: Mode = Mode.CENTERIZED,
) -> torch.Tensor:
    """compute the information gain of a set of features

    Args:
        features (torch.Tensor): features to compute information gain
        mean (torch.Tensor): mean of the features
        cov_another (torch.Tensor): covariance of the features

    Returns:
        torch.Tensor: information gain
    """
    features = features.double()
    mean = mean.double()
    cov_another = cov_another.double()

    diff = features - mean[None, :]
    if mode == Mode.CENTERIZED:
        out = torch.linalg.vector_norm(diff, ord=2, dim=-1) ** 2
    elif mode == Mode.WHITEN:
        out = torch.sum((diff @ cov_another) * diff, dim=1)
    else:
        raise ValueError()

    return out

@torch.no_grad()
def get_information_gain_2509(
    target_emb: torch.Tensor, 
    sample_emb: torch.Tensor,
    logit_scale: float | None = None,
    logit_bias: float | None = None,
    add_logz: bool = True,
    mode: Mode = Mode.MC
) -> torch.Tensor:
    """compute the information gain of a set of features

    Args:
        target_emb (torch.Tensor): features to compute information gain
        sample_emb (torch.Tensor): samples from another distribution

    Returns:
        torch.Tensor: information gain
    """
    print(f"sample_emb: min {sample_emb.min().item():.4f}, max {sample_emb.max().item():.4f}, mean {sample_emb.mean().item():.4f}, std {sample_emb.std().item():.4f}")
    print(f"target_emb: min {target_emb.min().item():.4f}, max {target_emb.max().item():.4f}, mean {target_emb.mean().item():.4f}, std {target_emb.std().item():.4f}")

    # cast to float32 for numerical stability
    target_emb = target_emb.double()
    sample_emb = sample_emb.double()

    # confirm that norm of sample embeddings is 1
    norms = torch.norm(sample_emb, dim=1)
    if torch.allclose(norms, torch.ones_like(norms), atol=1e-3):
        print("sample embeddings are normalized")
    else:
        print("sample embeddings are not normalized, normalizing")
        sample_emb = F.normalize(sample_emb, dim=1)
        target_emb = F.normalize(target_emb, dim=1)
        norms = torch.norm(sample_emb, dim=1)
        # assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3), f"norms: {norms}, torch.ones_like(norms): {torch.ones_like(norms)}"

    a = math.exp(logit_scale) if logit_scale is not None else 1.0
    b = logit_bias if logit_bias is not None else 0.0
    N = len(sample_emb)

    print(target_emb.shape, sample_emb.shape, a, b, N)
    # print(target_emb.dtype, sample_emb.dtype)

    print(f"sample_emb: min {sample_emb.min().item():.4f}, max {sample_emb.max().item():.4f}, mean {sample_emb.mean().item():.4f}, std {sample_emb.std().item():.4f}")
    print(f"target_emb: min {target_emb.min().item():.4f}, max {target_emb.max().item():.4f}, mean {target_emb.mean().item():.4f}, std {target_emb.std().item():.4f}")

    score = a * target_emb @ sample_emb.T
    if mode == Mode.REVERSE_MC:
        sumti = torch.sum(score, dim=-1)
        logsumexp = torch.logsumexp(score, dim=-1)
        kl = logsumexp - math.log(N) - (1 / N) * sumti
    elif mode == Mode.MC:
        softmax = F.softmax(score, dim=-1)
        s_softmax = torch.sum(softmax * score, dim=-1)
        logsumexp = torch.logsumexp(score, dim=-1)
        kl = s_softmax - (logsumexp - math.log(N))
    else:
        raise ValueError()

    # print statistics
    print(f"score: min {score.min().item():.4f}, max {score.max().item():.4f}, mean {score.mean().item():.4f}, std {score.std().item():.4f}")
    print(f"kl: min {kl.min().item():.4f}, max {kl.max().item():.4f}, mean {kl.mean().item():.4f}, std {kl.std().item():.4f}")

    return kl


def infogain_helper(
    mode: Mode,
    in_queue: mp.Queue,
    src_root: str,
    tgt_root: str,
    b32_stats_path: str,
    l14_stats_path: str,
    add_logz: bool = True,
) -> None:
    # ソース／ターゲット FS とプレフィックスを取得
    fs_src, src_prefix = fsspec.core.url_to_fs(src_root)
    fs_dst, dst_prefix = fsspec.core.url_to_fs(tgt_root)
    src_prefix = src_prefix.rstrip("/") + "/"
    dst_prefix = dst_prefix.rstrip("/") + "/"

    # stats 用 FS を取得してロード
    fs_b32, b32_url = fsspec.core.url_to_fs(b32_stats_path)
    fs_l14, l14_url = fsspec.core.url_to_fs(l14_stats_path)

    b32_txt_mean, b32_txt_cov = None, None
    b32_img_mean, b32_img_cov = None, None
    l14_txt_mean, l14_txt_cov = None, None
    l14_img_mean, l14_img_cov = None, None

    b32_txt_sample, b32_img_sample = None, None
    l14_txt_sample, l14_img_sample = None, None

    if is_2504(mode):
        with fs_b32.open(b32_url, "rb") as f:
            b32_stats = np.load(f)
            b32_txt_mean = torch.from_numpy(b32_stats["txt_mean"])
            b32_txt_cov  = torch.from_numpy(b32_stats["txt_cov"]).float()
            b32_img_mean = torch.from_numpy(b32_stats["img_mean"])
            b32_img_cov  = torch.from_numpy(b32_stats["img_cov"]).float()
        with fs_l14.open(l14_url, "rb") as f:
            l14_stats = np.load(f)
            l14_txt_mean = torch.from_numpy(l14_stats["txt_mean"])
            l14_txt_cov  = torch.from_numpy(l14_stats["txt_cov"]).float()
            l14_img_mean = torch.from_numpy(l14_stats["img_mean"])
            l14_img_cov  = torch.from_numpy(l14_stats["img_cov"]).float()
    elif is_2509(mode):
        with fs_b32.open(b32_url, "rb") as f:
            b32_stats = np.load(f)
            b32_txt_sample = torch.from_numpy(b32_stats["txt_embeddings"])
            b32_img_sample = torch.from_numpy(b32_stats["img_embeddings"])
        with fs_l14.open(l14_url, "rb") as f:
            l14_stats = np.load(f)
            l14_txt_sample = torch.from_numpy(l14_stats["txt_embeddings"])
            l14_img_sample = torch.from_numpy(l14_stats["img_embeddings"])
    else:
        raise ValueError()

    while True:
        try:
            fs, rel_path = in_queue.get(timeout=1)
        except Empty:
            return

        # parquet 読み込み
        df = pd.read_parquet(f"{src_prefix}{rel_path}.parquet", filesystem=fs)

        # npz 埋め込みを一度だけ読み込み
        with fs.open(f"{src_prefix}{rel_path}.npz", "rb") as f:
            data = np.load(f)
            b32_txt_emb = torch.from_numpy(data["b32_txt"])
            b32_img_emb = torch.from_numpy(data["b32_img"])
            l14_txt_emb = torch.from_numpy(data["l14_txt"])
            l14_img_emb = torch.from_numpy(data["l14_img"])

        # 次元チェック
        D32 = b32_txt_mean.shape[0] if b32_txt_mean is not None else b32_img_sample.shape[1]
        assert all(e.shape[1] == D32 for e in (b32_txt_emb, b32_img_emb))
        D14 = l14_txt_mean.shape[0] if l14_txt_mean is not None else l14_img_sample.shape[1]
        assert all(e.shape[1] == D14 for e in (l14_txt_emb, l14_img_emb))

        # 情報利得計算
        if is_2504(mode):
            assert None not in (
                b32_txt_mean,
                b32_txt_cov,
                b32_img_mean,
                b32_img_cov,
                l14_txt_mean,
                l14_txt_cov,
                l14_img_mean,
                l14_img_cov,
            ), "stats not loaded properly"
            df["ig_b32_txt"] = get_information_gain_2504(b32_txt_emb, b32_txt_mean, b32_img_cov, mode).cpu().numpy()
            df["ig_b32_img"] = get_information_gain_2504(b32_img_emb, b32_img_mean, b32_txt_cov, mode).cpu().numpy()
            df["ig_l14_txt"] = get_information_gain_2504(l14_txt_emb, l14_txt_mean, l14_img_cov, mode).cpu().numpy()
            df["ig_l14_img"] = get_information_gain_2504(l14_img_emb, l14_img_mean, l14_txt_cov, mode).cpu().numpy()
        elif is_2509(mode):
            assert None not in (
                b32_txt_sample,
                b32_img_sample,
                l14_txt_sample,
                l14_img_sample,
            ), "stats not loaded properly"
            df["ig_b32_txt"] = get_information_gain_2509(b32_txt_emb, b32_img_sample, b32_logit_scale, b32_logit_bias, add_logz, mode).cpu().numpy()
            df["ig_b32_img"] = get_information_gain_2509(b32_img_emb, b32_txt_sample, b32_logit_scale, b32_logit_bias, add_logz, mode).cpu().numpy()
            df["ig_l14_txt"] = get_information_gain_2509(l14_txt_emb, l14_img_sample, l14_logit_scale, l14_logit_bias, add_logz, mode).cpu().numpy()
            df["ig_l14_img"] = get_information_gain_2509(l14_img_emb, l14_txt_sample, l14_logit_scale, l14_logit_bias, add_logz, mode).cpu().numpy()
        else:
            raise ValueError()
        # parquet 保存
        out_parquet = f"{dst_prefix}{rel_path}.parquet"
        out_dir = os.path.dirname(out_parquet)
        if not fs_dst.exists(out_dir):
            fs_dst.mkdirs(out_dir, exist_ok=True)
        df.to_parquet(out_parquet, filesystem=fs_dst)

        # 同階層の npz をコピー
        src_npz = f"{src_prefix}{rel_path}.npz"
        dst_npz = f"{dst_prefix}{rel_path}.npz"
        dst_npz_dir = os.path.dirname(dst_npz)
        if not fs_dst.exists(dst_npz_dir):
            fs_dst.mkdirs(dst_npz_dir, exist_ok=True)
        with fs_src.open(src_npz, "rb") as fr, fs_dst.open(dst_npz, "wb") as fw:
            fw.write(fr.read())


def main(
    mode: Mode,
    metadata_dir_path: str,
    metadata_save_dir: str,
    num_gpus: int,
    b32_stats_path: Union[str, None] = None,
    l14_stats_path: Union[str, None] = None,
    num_workers: Union[int, None] = None,
    add_logz: bool = True,
) -> None:
    """image based filter from the datacomp paper

    Args:
        metadata_dir_path (str): directory where metadata is stored
        num_gpus (int): number of gpu workers, each of which processes parquet, npy pairs
        arch (Union[str, None], optional): kind of features for clip filtering. Defaults to None.
        num_workers (Union[int, None], optional): number of cpu works used to load metadata to compute threshold. Defaults to None.

    Raises:
        RuntimeError: raises in case of a queue mishap, should not happen

    Returns:
        np.ndarray: array of uids
    """
    fs, base_url = fsspec.core.url_to_fs(metadata_dir_path)
    prefix = base_url.rstrip("/") + "/"

    send_queue = mp.Queue()
    for path in fs.ls(base_url):
        if path.endswith(".parquet"):
            rel = path[len(prefix):-len(".parquet")]
            send_queue.put((fs, rel))

    os.makedirs(metadata_save_dir, exist_ok=True)

    procs = []
    for _ in range(num_gpus):
        p = mp.Process(
            target=infogain_helper,
            kwargs=dict(
                mode=mode,
                in_queue=send_queue,
                src_root=metadata_dir_path,
                tgt_root=metadata_save_dir,
                b32_stats_path=b32_stats_path,
                l14_stats_path=l14_stats_path,
                add_logz=add_logz,
            ),
        )
        p.start()
        procs.append(p)
        time.sleep(0.05)

    for p in procs:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["MC", "REVERSE_MC", "CENTERIZED", "WHITEN"],
        default="MC",
        help="which infogain mode to run",
    )
    parser.add_argument(
        "--metadata_dir",
        type=str,
        help="directory (local or cloud) containing parquet, npz metadata",
    )
    parser.add_argument("--save_path", type=str, help="local path to output centroids")
    parser.add_argument(
        "--b32_stats_path",
        type=str,
        help="path to b32 stats npz file",
        default=None,
    )
    parser.add_argument(
        "--l14_stats_path",
        type=str,
        help="path to l14 stats npz file",
        default=None,
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=torch.cuda.device_count(),
        help="number of gpu workers, each of which processes parquet, npy pairs",
    )
    # add_logz is true by default
    parser.add_argument(
        "--disable_logz",
        action="store_true",
        help="whether to disable adding logz term in 2509 infogain",
    )
    args = parser.parse_args()

    if args.mode == "MC":
        mode = Mode.MC
    elif args.mode == "REVERSE_MC":
        mode = Mode.REVERSE_MC
    elif args.mode == "CENTERIZED":
        mode = Mode.CENTERIZED
    elif args.mode == "WHITEN":
        mode = Mode.WHITEN

    main(
        mode=mode,
        metadata_dir_path=args.metadata_dir,
        metadata_save_dir=args.save_path,
        num_gpus=args.num_gpus,
        b32_stats_path=args.b32_stats_path,
        l14_stats_path=args.l14_stats_path,
        add_logz=not args.disable_logz,
    )