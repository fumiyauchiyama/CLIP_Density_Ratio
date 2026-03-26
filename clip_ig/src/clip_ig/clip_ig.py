from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Literal, Union, Tuple
import math
from logging import getLogger

import numpy as np
import torch
from open_clip import (
    CLIP,
    CustomTextCLIP,
    create_model_from_pretrained,
    get_tokenizer,
    create_model_and_transforms,
)
from open_clip.tokenizer import HFTokenizer, SigLipTokenizer, SimpleTokenizer
from PIL import Image
from torch.nn import Module
from torch.nn import functional as F
from torchtyping import TensorType as TT  # type: ignore
from torchvision.transforms import Compose

logger = getLogger(__name__)


class IGMode(Enum):
    WHITEN = "whiten"
    CENTERIZED = "centerized"
    RAW_NORM = "raw_norm"
    REVERSE_MC = "reverse_mc"
    MC = "mc"


class ModelType(Enum):
    CLIP = "clip"
    SIGLIP = "siglip"

class TauBLoc(Enum):
    ALL = "all"
    LOG_Z = "log_z"

@dataclass
class SiglipConfig:
    use_tau_b: bool = False
    tau_b_loc: TauBLoc = TauBLoc.ALL

@dataclass
class ModelConfig:
    model_name: str = "ViT-B-16"
    pretrained: str | None = "openai"
    model_type: ModelType | None = None
    device: str = "cuda"
    logit_scale: float | None = None
    logit_bias: float | None = None
    siglip_config: SiglipConfig = field(default_factory=SiglipConfig)


class Predictor:
    def __init__(
        self,
        model: Union[CLIP, CustomTextCLIP],
        model_type: ModelType | None = None,
        device: str = "cuda",
        logit_scale: float | None = None,
        logit_bias: float | None = None,
        siglip_config: SiglipConfig = field(default_factory=SiglipConfig),
    ) -> None:
        self.model = model
        assert model_type is not None
        self.model_type = model_type
        self.siglip_config = siglip_config
        self.device = device

        # Unnormalized image and text embeddings from the sample data
        self.sample_image_emb: TT["num_sample", "hidden_dim"] | None = (
            None  # noqa: F821
        )
        self.sample_text_emb: TT["num_sample", "hidden_dim"] | None = None  # noqa: F821

        # Expected value and covariance of image and text embeddings
        self.image_mean: TT["hidden_dim"] | None = None  # noqa: F821
        self.image_cov: TT["hidden_dim", "hidden_dim"] | None = None  # noqa: F821
        self.text_mean: TT["hidden_dim"] | None = None  # noqa: F821
        self.text_cov: TT["hidden_dim", "hidden_dim"] | None = None  # noqa: F821

        self.model.to(device)
        self.model.eval()

        # logit scale, logit bias
        self.a = (
            self.model.logit_scale.exp()
            if self.model.logit_scale is not None
            else torch.tensor(1.0)
        )
        self.b = (
            self.model.logit_bias
            if self.model.logit_bias is not None
            else torch.tensor(0.0)
        )

        if logit_scale is not None:
            self.a = torch.tensor(logit_scale)
        if logit_bias is not None:
            self.b = torch.tensor(logit_bias)

        logger.info(f"logit scale: {float(self.a)}, logit bias: {float(self.b)}")

    def encode_image(
        self,
        image: TT["num_batch", "num_channels", "height", "width"],
        normalize: bool = False,
        bsz: int = 512,
    ) -> TT["num_batch", "hidden_dim"]:  # noqa: F821
        with torch.no_grad():
            output = []
            for i in range(0, image.shape[0], bsz):
                batch = image[i : i + bsz].to(self.device)
                output.append(self.model.encode_image(batch, normalize=normalize))
            output = torch.cat(output, dim=0)
            return output  # type: ignore
            # return self.model.encode_image(image, normalize=normalize)  # type: ignore

    def encode_text(
        self,
        text: TT["num_batch", "text_length"],
        normalize: bool = False,
    ) -> TT["num_batch", "hidden_dim"]:
        with torch.no_grad():
            output = []
            for i in range(0, text.shape[0], 512):
                batch = text[i : i + 512].to(self.device)
                output.append(self.model.encode_text(batch, normalize=normalize))
            output = torch.cat(output, dim=0)
            return output  # type: ignore
            # return self.model.encode_text(text, normalize=normalize)  # type: ignore

    def encode_sample_image(
        self, sample_image: TT["num_batch", "num_channels", "height", "width"]
    ) -> None:
        self.sample_image_emb = self.encode_image(sample_image)

    def encode_sample_text(self, sample_text: TT["num_batch", "text_length"]) -> None:
        self.sample_text_emb = self.encode_text(sample_text)

    def _calc_mean(
        self, emb: TT["num_batch", "hidden_dim"], normalize: bool = True
    ) -> TT["hidden_dim"]:
        if normalize:
            emb = F.normalize(emb, dim=-1)
        mean = torch.mean(emb, dim=0)
        assert mean.shape == (emb.shape[1],)
        return mean

    def calc_image_mean(self, normalize=True) -> None:
        assert self.sample_image_emb is not None
        self.image_mean = self._calc_mean(self.sample_image_emb, normalize=normalize)

    def calc_text_mean(self, normalize=True) -> None:
        assert self.sample_text_emb is not None
        self.text_mean = self._calc_mean(self.sample_text_emb, normalize=normalize)

    def _calc_cov(
        self,
        emb: TT["num_sample", "hidden_dim"],  # noqa: F821
        mean: TT["hidden_dim"],  # noqa: F821
        normalize: bool = True,
    ) -> TT["hidden_dim", "hidden_dim"]:  # noqa: F821
        if normalize:
            emb = F.normalize(emb, dim=-1)
        X: TT["num_sample", "hidden_dim"] = emb - mean  # (N, D)
        cov = X.T @ X  # (D, N) @ (N, D) -> (D, D)
        assert cov.shape == (emb.shape[1], emb.shape[1])
        return cov

    def calc_image_cov(self, normalize=True) -> None:
        assert self.sample_image_emb is not None

        if self.image_mean is None:
            self.calc_image_mean(normalize=normalize)

        self.image_cov = self._calc_cov(
            self.sample_image_emb, self.image_mean, normalize=normalize
        )

    def calc_text_cov(self, normalize=True) -> None:
        assert self.sample_text_emb is not None

        if self.text_mean is None:
            self.calc_text_mean(normalize=normalize)

        self.text_cov = self._calc_cov(
            self.sample_text_emb, self.text_mean, normalize=normalize
        )

    def _information_gain(
        self,
        target_emb: TT["num_batch", "hidden_dim"],  # noqa: F821
        mode: IGMode = IGMode.WHITEN,
        mean: TT["hidden_dim"] | None = None,
        cov_another: TT["hidden_dim", "hidden_dim"] | None = None,  # noqa: F821
        normalize: bool = True,
    ) -> TT["num_batch"]:
        assert (
            len(target_emb.shape) == 2
        ), f"target_emb should be [num_batch, hidden_dim] tensor, but got {target_emb.shape}"
        output: TT["num_batch"] | None = None

        if normalize:
            target_emb = F.normalize(target_emb, dim=-1)

        if mode == IGMode.WHITEN:
            assert mean is not None
            assert cov_another is not None
            cnt_target_emb: TT["num_batch", "hidden_dim"] = target_emb - mean.unsqueeze(
                0
            )
            # Compute x^T * text_cov * x for each batch via batch matmul
            cov_expanded: TT["num_batch", "hidden_dim", "hidden_dim"] = (
                cov_another.unsqueeze(0).expand(cnt_target_emb.size(0), -1, -1)
            )
            mid: TT["num_batch", 1, 1] = torch.bmm(
                torch.bmm(cnt_target_emb.unsqueeze(1), cov_expanded),
                cnt_target_emb.unsqueeze(2),
            )
            output = mid.squeeze(2).squeeze(1)  # type: ignore
        elif mode == IGMode.CENTERIZED:
            assert mean is not None
            cnt_target_emb: TT["num_batch", "hidden_dim"] = target_emb - mean.unsqueeze(
                0
            )
            output = (
                torch.linalg.vector_norm(cnt_target_emb, ord=2, dim=-1) ** 2
            )  # type: ignore
        elif mode == IGMode.RAW_NORM:
            output = (
                torch.linalg.vector_norm(target_emb, ord=2, dim=-1) ** 2
            )  # type: ignore
        else:
            raise ValueError(
                f"mode should be 'RAW_NORM', 'CENTERIZED' or 'WHITEN', but got {mode}"
            )

        assert output is not None

        return (self.a**2) * output

    def _information_gain_mc(
        self,
        target_emb: TT["num_batch", "hidden_dim"],  # noqa: F821
        sample_emb: TT["num_sample", "hidden_dim"],  # noqa: F821
        mode: IGMode = IGMode.MC,
    ) -> TT["num_batch"] | Tuple[TT["num_batch"], TT["num_batch"]]:
        assert (
            len(target_emb.shape) == 2
        ), f"target_emb should be [num_batch, hidden_dim] tensor, but got {target_emb.shape}"
        assert (
            len(sample_emb.shape) == 2
        ), f"sample_emb should be [num_sample, hidden_dim] tensor, but got {sample_emb.shape}"

        # normalize
        target_emb = F.normalize(target_emb, dim=-1)
        sample_emb = F.normalize(sample_emb, dim=-1)

        # logit scale, logit bias
        N = len(sample_emb)

        score: TT["num_batch", "num_sample"] = (
            self.a * target_emb @ sample_emb.T
        )  # b is canceled out

        tau = N - 1
        kl: TT["num_batch"] | None = None

        if mode == IGMode.REVERSE_MC:
            meanti = torch.mean(score, dim=-1)
            if self.model_type == ModelType.SIGLIP and self.siglip_config.use_tau_b:
                kl = - math.log(tau) - self.b - meanti
            else:
                logsumexp: TT["num_batch"] = torch.logsumexp(score, dim=-1)
                kl = logsumexp - math.log(N) - meanti
        elif mode == IGMode.MC:
            if self.model_type == ModelType.SIGLIP and self.siglip_config.use_tau_b:
                if self.siglip_config.tau_b_loc == TauBLoc.ALL:
                    score_f64 = score.to(torch.float64)
                    exps = score_f64 * score_f64.exp()
                    b_f64 = self.b.to(torch.float64)
                    kl = tau * b_f64.exp() * torch.mean(exps, dim=-1) + math.log(tau) + b_f64
                elif self.siglip_config.tau_b_loc == TauBLoc.LOG_Z:
                    softmax = F.softmax(score, dim=-1)
                    s_softmax = torch.sum(softmax * score, dim=-1)
                    kl = s_softmax + math.log(tau) + self.b
                else:
                    raise ValueError("Unknown TauBLoc")
            else:
                softmax = F.softmax(score, dim=-1)
                s_softmax = torch.sum(softmax * score, dim=-1)
                logsumexp: TT["num_batch"] = torch.logsumexp(score, dim=-1)
                kl = s_softmax - (logsumexp - math.log(N))
        else:
            raise ValueError(f"mode should be 'MC' or 'REVERSE_MC', but got {mode}")

        assert kl is not None

        return kl

    def ig_image_emb(
        self,
        target_image: TT["num_batch", "hidden_dim"],  # noqa: F821
        mode: IGMode = IGMode.MC,
        normalize: bool = True,
    ) -> TT["num_batch"] | Tuple[TT["num_batch"], TT["num_batch"]]:
        """
        Calculate the information gain of the image embeddings
        Args:
            target_image (TT["num_batch", "hidden_dim"]):
                Image Embedding for conditioning text prediction logits
            mode (IGMode):
                IGMode
            normalize (bool):
                set True if you want to normalize embeddings during calculation of norm-based metrics.
                Note that expected value andcovariance matrix are already calculated.
                Please make suke they are also normalized if you set normalize=True.
        Returns:
            TT["num_batch"]: Information gain of the image embeddings
        """

        if mode == IGMode.REVERSE_MC or mode == IGMode.MC:
            return self._information_gain_mc(
                target_image,
                sample_emb=self.sample_text_emb,
                mode=mode,
            )
        else:
            return self._information_gain(
                target_image,
                mode=mode,
                mean=self.image_mean,
                cov_another=self.text_cov,
                normalize=normalize,
            )

    def ig_text_emb(
        self,
        target_text: TT["num_batch", "hidden_dim"],  # noqa: F821
        mode: IGMode = IGMode.MC,
        normalize: bool = True,
    ) -> TT["num_batch"] | Tuple[TT["num_batch"], TT["num_batch"]]:
        """
        Calculate the information gain of the image embeddings
        Args:
            target_text (TT["num_batch", "hidden_dim"]):
                Text Embedding for conditioning image prediction logits
            mode (IGMode):
                IGMode
            normalize (bool):
                set True if you want to normalize embeddings during calculation of norm-based metrics.
                Note that expected value andcovariance matrix are already calculated.
                Please make suke they are also normalized if you set normalize=True.
        Returns:
            TT["num_batch"]: Information gain of the text embeddings
        """

        if mode == IGMode.REVERSE_MC or mode == IGMode.MC:
            return self._information_gain_mc(
                target_text,
                sample_emb=self.sample_image_emb,
                mode=mode,
            )
        else:
            return self._information_gain(
                target_text,
                mode=mode,
                mean=self.text_mean,
                cov_another=self.image_cov,
                normalize=normalize,
            )

    def ig_image(
        self,
        target_image: TT["num_batch", "num_channels", "height", "width"],
        mode: IGMode = IGMode.MC,
        normalize: bool = True,
    ) -> TT["num_batch"] | Tuple[TT["num_batch"], TT["num_batch"]]:
        with torch.no_grad():
            target_image_emb = self.encode_image(target_image)
            return self.ig_image_emb(
                target_image_emb,
                mode=mode,
                normalize=normalize,
            )

    def ig_text(
        self,
        target_text: TT["num_batch", "text_length"],
        mode: IGMode = IGMode.MC,
        normalize: bool = True,
    ) -> TT["num_batch"] | Tuple[TT["num_batch"], TT["num_batch"]]:
        with torch.no_grad():
            target_text_emb = self.encode_text(target_text)
            return self.ig_text_emb(
                target_text_emb,
                mode=mode,
                normalize=normalize,
            )

    def reset_text_embedding(self) -> None:
        self.sample_text_emb = None

    def reset_image_embedding(self) -> None:
        self.sample_image_emb = None

    def reset(self) -> None:
        self.sample_image_emb = None
        self.sample_text_emb = None
        self.image_mean = None
        self.image_cov = None
        self.text_mean = None
        self.text_cov = None


def get_predictor(
    model: Union[CLIP, CustomTextCLIP],
    model_type: ModelType,
    device: str = "cuda",
    siglip_config: SiglipConfig | None = None,
) -> Predictor:

    predictor = Predictor(
        model,
        model_type=model_type, 
        device=device,
        siglip_config=siglip_config,
        )

    return predictor


def get_predictor_from_open_clip(
    model_cfg: ModelConfig,
) -> Predictor:
    device = model_cfg.device
    model_kwargs: Dict[str, Union[float, str]] = {}

    if model_cfg.pretrained is not None:
        vl_model, image_processor = create_model_from_pretrained(
            model_cfg.model_name,
            pretrained=model_cfg.pretrained,
            device=device,
            **model_kwargs,
        )  # type: ignore
    else:
        print(f"Pretrained model not specified. Loading from scratch.")
        vl_model, _, image_processor = create_model_and_transforms(
            model_cfg.model_name,
            device=device,
            **model_kwargs,
        )

    tokenizer = get_tokenizer(model_cfg.model_name)

    assert isinstance(
        vl_model, (CLIP, CustomTextCLIP)
    ), f"model should be CLIP, but got {type(vl_model)}"
    assert isinstance(
        image_processor, Compose
    ), f"image_processor should be Compose, but got {type(image_processor)}"
    assert isinstance(
        tokenizer, (HFTokenizer, SigLipTokenizer, SimpleTokenizer)
    ), f"tokenizer should be HFTokenizer or SigLipTokenizer or SimpleTokenizer, but got {type(tokenizer)}"

    predictor = Predictor(
        vl_model,
        model_type=model_cfg.model_type,
        device=device,
        logit_scale=model_cfg.logit_scale,
        logit_bias=model_cfg.logit_bias,
        siglip_config=model_cfg.siglip_config,
    )

    return predictor, image_processor, tokenizer
