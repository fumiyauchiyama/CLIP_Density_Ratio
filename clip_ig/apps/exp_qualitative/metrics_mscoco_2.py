from enum import Enum
import os
import time
import json
from dataclasses import dataclass, field
from io import BytesIO
from random import sample
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import webdataset as wds
import numpy as np
from logging import getLogger, INFO, basicConfig

import requests
import torch
from clip_ig import IGMode, ModelConfig, Predictor, get_predictor_from_open_clip
from datasets import load_dataset, load_from_disk
from omegaconf import OmegaConf
from PIL import Image
from sklearn.metrics import precision_recall_curve
from torchvision import transforms
from tqdm import tqdm
import csv

from .utils import dump_config, save_csv

basicConfig(level=INFO)
logger = getLogger(__name__)

class Modal(Enum):
    TXT = "txt"
    IMG = "img"


@dataclass
class DatasetConfig:
    dataset_name: str = "path-to-MSCOCO2017"
    n_samples: int = 100000000000
    image_url_key: str = "url"
    text_key: str = "caption"


@dataclass
class EvalConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    normalize_norm_metrics: bool = True
    mode: IGMode = IGMode.MC


def save_sample_per_bin(
    samples: Union[List[str], List[Image.Image]],
    metrics: torch.Tensor,
    save_dir: str,
    bin_size: int = 1000,
    n_sample_per_bin: int = 500,
) -> None:
    assert len(samples) == len(metrics)
    assert bin_size >= n_sample_per_bin
    # sort samples w.r.t. metrics
    sorted_indices = torch.argsort(metrics)
    samples = [samples[i] for i in sorted_indices]
    metrics = metrics[sorted_indices]

    os.makedirs(save_dir, exist_ok=True)
    for i in range(0, len(samples), bin_size):
        batch = samples[i : i + bin_size]
        # randomly pickup sample data from batch and save them
        if len(batch) > n_sample_per_bin:
            sampled_data = sample(batch, n_sample_per_bin)
        else:
            sampled_data = batch
        if isinstance(sampled_data[0], Image.Image):
            sample_per_bin_save_dir = os.path.join(save_dir, f"bin_{i // bin_size}")
            os.makedirs(sample_per_bin_save_dir, exist_ok=True)
            for j, data in enumerate(sampled_data):
                data.save(
                    os.path.join(
                        sample_per_bin_save_dir, f"bin_{i // bin_size}_image_{j}.png"
                    )
                )
        else:
            sample_data_dict = {"text": sampled_data}
            save_csv(
                sample_data_dict,
                os.path.join(save_dir, f"bin_{i // bin_size}_text.csv"),
            )


def extract_top_bottom_samples(
    eval_data: Union[List[str], List[Image.Image]],
    sample_data: Union[List[Image.Image], List[str]],
    predictor: Predictor,
    processor: Optional[Callable[[Any], Any]] = None,
    tokenizer: Optional[Callable[[Any], Any]] = None,
    normalize_norm_metrics: bool = True,
    mode=IGMode.MC,
    k: int = 30,
    save_dir: str = "./output",
    sample_image_id = None,
    img_id_caps_dict = None,
):
    # Determine which modality and mode to calc metrics
    modal: Modal | None = None
    if isinstance(eval_data[0], str):
        modal = Modal.TXT
    elif isinstance(eval_data[0], Image.Image):
        modal = Modal.IMG
    else:
        raise ValueError()

    # Prepare input data
    if modal == Modal.TXT:
        assert tokenizer is not None
        input_data = tokenizer(eval_data).to(predictor.device)
        calc_fn = predictor.ig_text
    elif modal == Modal.IMG:
        assert processor is not None
        input_list = [processor(image).to(predictor.device) for image in eval_data]
        input_data = torch.stack(input_list)
        assert (
            len(input_data.shape) == 4
        ), f"input should be [num_batch, num_cahnnels, height, width] tensor, but got {input_data.shape}"
        calc_fn = predictor.ig_image
    else:
        raise ValueError()

    # Calc metrics and save them to {modal.value}_{mode.value}.npz
    metrics_dict: dict[str, Any] = dict()
    ig = calc_fn(
        input_data,
        mode=mode,
        normalize=normalize_norm_metrics,
    )

    logger.info(f"{modal.value} KLs:")
    logger.info(f"Mean: {torch.mean(ig).item()}, Std: {torch.std(ig).item()}")

    metrics_dict.update(
        {
            "ig": ig.cpu().numpy(),
            "a": float(predictor.a),
            "b": float(predictor.b),
            "tau": len(predictor.sample_image_emb) - 1,
        }
    )

    if modal == Modal.TXT:
        metrics_dict.update({"caption": np.array(eval_data),})

    if sample_image_id is not None:
        assert modal == Modal.IMG
        assert img_id_caps_dict is not None
        metrics_dict.update({"id": np.array(sample_image_id),})
        np.savez(
            os.path.join(save_dir, f"{modal.value}_{mode.value}_caption.npz"),
            **img_id_caps_dict,
        )

    np.savez(
        os.path.join(save_dir, f"{modal.value}_{mode.value}.npz"),
        **metrics_dict,
    )
    
    # Save each samples per bin
    save_sample_per_bin(
        samples=eval_data,
        metrics=ig,
        save_dir=os.path.join(save_dir, f"sample_{modal.value}_per_bin"),
        bin_size=len(eval_data) // 10,
        n_sample_per_bin=len(eval_data) // 10,
    )

    # Save top and bottom KL img/txt and corresponding txt/img
    top_k_values, top_k_indices = torch.topk(ig, k)
    top_k_values = top_k_values.cpu().numpy()
    top_k_indices = top_k_indices.cpu().numpy()
    top_k_samples = [eval_data[i] for i in top_k_indices]

    bottom_k_values, bottom_k_indices = torch.topk(ig, k, largest=False)
    bottom_k_values = bottom_k_values.cpu().numpy()
    bottom_k_indices = bottom_k_indices.cpu().numpy()
    bottom_k_samples = [eval_data[i] for i in bottom_k_indices]

    top_value_dict = {
        "top_k_values": top_k_values,
    }

    bottom_value_dict = {
        "bottom_k_values": bottom_k_values,
    }

    if modal == Modal.TXT:
        top_value_dict["top_k_samples"] = top_k_samples
        bottom_value_dict["bottom_k_samples"] = bottom_k_samples

    elif modal == Modal.IMG:
        # save top and bottom ig images
        image_save_dir = os.path.join(save_dir, "ig_images")
        os.makedirs(image_save_dir, exist_ok=True)
        for i, image in enumerate(top_k_samples):
            image.save(os.path.join(image_save_dir, f"image_top_{i}.png"))
        for i, image in enumerate(bottom_k_samples):
            image.save(os.path.join(image_save_dir, f"image_bottom_{i}.png"))
    else:
        raise ValueError()

    save_csv(
        top_value_dict,
        os.path.join(save_dir, f"{modal.value}_top_k_info.csv"),
    )
    save_csv(
        bottom_value_dict,
        os.path.join(save_dir, f"{modal.value}_bottom_k_info.csv"),
    )


def main(
    cfg: EvalConfig,
):
    save_dir: str = "./output"
    date_str = time.strftime("%Y%m%d")
    time_str = time.strftime("%Y%m%d-%H%M%S") + "sample_mscoco2"
    save_dir = os.path.join(save_dir, date_str, time_str)
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"save dir: {save_dir}")

    config_path = os.path.join(save_dir, f"{time_str}_vis_config.yaml")
    dump_config(cfg, config_path)

    # Prepare sample dataset to calc metrics

    with open(os.path.join(cfg.dataset.dataset_name, "annotations", "captions_val2017.json"),"r") as f:
        annotations = json.load(f)['annotations']

    
    sample_text = []
    sample_img_id_set = set()
    img_id_caps_dict = dict()
    count = 0
    for sample in annotations:
        if isinstance(sample["caption"], bytes):
            t = sample["caption"].decode("utf-8")
            logger.debug(f"decoded from utf-8: {t}")
        elif isinstance(sample["caption"], str):
            t = sample["caption"]
        else:
            raise ValueError(
                f"sample['caption'] should be either bytes or str: {type(sample['caption'])}"
            )
        sample_text.append(t)
        sample_img_id_set.add(sample["image_id"])
        entry = img_id_caps_dict.get(str(sample["image_id"]))
        if entry is None:
            img_id_caps_dict[str(sample["image_id"])] = [t]
        else:
            img_id_caps_dict[str(sample["image_id"])].append(t)
        count += 1
        if count >= cfg.dataset.n_samples:
            break

    img_id_caps_dict_len = [len(img_id_caps_dict[e]) for e in img_id_caps_dict.keys()]
    print(sum(img_id_caps_dict_len)/len(img_id_caps_dict_len))

    sample_image = []
    sample_image_id = []

    for sample_id in sample_img_id_set:
        img_path = os.path.join(cfg.dataset.dataset_name, "images", "val2017", f"{sample_id:012d}.jpg")
        sample_image.append(Image.open(img_path))
        sample_image_id.append(sample_id)

    logger.info("Sample data loaded")

    # Encode samples and compute statistics

    predictor, processor, tokenizer = get_predictor_from_open_clip(cfg.model)
    device = cfg.model.device

    sample_image_input_list = [processor(image).to(predictor.device) for image in sample_image]
    sample_image_input = torch.stack(sample_image_input_list).to(device)
    sample_text_input = tokenizer(sample_text).to(device)

    predictor.encode_sample_image(sample_image_input)
    predictor.encode_sample_text(sample_text_input)

    if cfg.mode not in (IGMode.MC, IGMode.REVERSE_MC):
        # For norm-based metrics
        predictor.calc_image_mean(normalize=cfg.normalize_norm_metrics)
        predictor.calc_text_mean(normalize=cfg.normalize_norm_metrics)
        predictor.calc_image_cov(normalize=cfg.normalize_norm_metrics)
        predictor.calc_text_cov(normalize=cfg.normalize_norm_metrics)

    extract_top_bottom_samples(
        sample_text,
        sample_image,
        predictor,
        processor=None,
        tokenizer=tokenizer,
        normalize_norm_metrics=cfg.normalize_norm_metrics,
        save_dir=save_dir,
        mode=cfg.mode,
    )

    extract_top_bottom_samples(
        sample_image,
        sample_text,
        predictor,
        processor=processor,
        tokenizer=None,
        normalize_norm_metrics=cfg.normalize_norm_metrics,
        save_dir=save_dir,
        mode=cfg.mode,
        sample_image_id=sample_image_id,
        img_id_caps_dict=img_id_caps_dict,
    )


if __name__ == "__main__":
    cli_args = OmegaConf.from_cli()
    if "config" in cli_args.keys():
        file_cfg = OmegaConf.load(cli_args.config)
        # We remove 'config' attribute from config as the underlying DataClass does not have it
        del cli_args.config
        cli_args = OmegaConf.merge(file_cfg, cli_args)
    default_config = OmegaConf.structured(EvalConfig)
    config = OmegaConf.merge(default_config, cli_args)
    config = OmegaConf.to_object(config)
    main(config)
