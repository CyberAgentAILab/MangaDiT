import torch
from typing import Optional, Union, List, Tuple
from diffusers.pipelines import FluxPipeline
from PIL import Image, ImageFilter
import numpy as np
import cv2

from .pipeline_tools import encode_images

condition_dict = {
    "lineart": 0,
    "reference": 1,
}


class Condition(object):
    def __init__(
        self,
        condition_type: str,
        raw_img: Union[Image.Image, torch.Tensor] = None,
        condition: Union[Image.Image, torch.Tensor] = None,
        mask=None,
        position_delta=None,
        position_scale=1.0,
    ) -> None:
        self.condition_type = condition_type
        assert raw_img is not None or condition is not None
        if raw_img is not None:
            self.condition = self.get_condition(condition_type, raw_img)
        else:
            self.condition = condition
        self.position_delta = position_delta
        self.position_scale = position_scale
        # TODO: Add mask support
        assert mask is None, "Mask not supported yet"

    def get_condition(
        self, condition_type: str, raw_img: Union[Image.Image, torch.Tensor]
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Returns the condition image.
        """
        if condition_type == "lineart":
            print('get condition line')
            return raw_img.convert("L").convert("RGB")
        elif condition_type == "reference":
            print('get condition ref')
            return raw_img.convert("RGB")
        
        return self.condition

    @property
    def type_id(self) -> int:
        """
        Returns the type id of the condition.
        """
        return condition_dict[self.condition_type]

    @classmethod
    def get_type_id(cls, condition_type: str) -> int:
        """
        Returns the type id of the condition.
        """
        return condition_dict[condition_type]

    def encode(self, pipe: FluxPipeline) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Encodes the condition into tokens, ids and type_id.
        """
        if self.condition_type in [
            "lineart",
            "reference",
        ]:
            tokens, ids = encode_images(pipe, self.condition)
        else:
            raise NotImplementedError(
                f"Condition type {self.condition_type} not implemented"
            )
        if self.position_delta is None and self.condition_type == "subject":
            self.position_delta = [0, -self.condition.size[0] // 16]
        if self.position_delta is not None:
            ids[:, 1] += self.position_delta[0]
            ids[:, 2] += self.position_delta[1]
        if self.position_scale != 1.0:
            scale_bias = (self.position_scale - 1.0) / 2
            ids[:, 1] *= self.position_scale
            ids[:, 2] *= self.position_scale
            ids[:, 1] += scale_bias
            ids[:, 2] += scale_bias
        type_id = torch.ones_like(ids[:, :1]) * self.type_id
        return tokens, ids, type_id
