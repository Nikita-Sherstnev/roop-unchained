from typing import Any, List, Callable
import cv2
import numpy as np
import torch
import torch.nn.functional as F

import roop.globals
from roop.typing import Face, Frame, FaceSet
from roop.utilities import resolve_relative_path
from roop.models.gpen import FaceGAN


class Enhance_GPEN():

    model_gpen = None
    name = None
    devicename = None

    processorname = 'gpen'
    type = 'enhance'


    def Initialize(self, devicename):
        if self.model_gpen is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model_gpen = FaceGAN(base_dir='models', in_size=512, out_size=None, model='GPEN-BFR-512', channel_multiplier=2,
                      narrow=1, key=None, device=self.device)
            self.devicename = devicename

        # self.name = self.model_gpen.get_inputs()[0].name

    def Run(self, source_faceset: FaceSet, target_face: Face, temp_frame: Frame) -> Frame:
        input_size = temp_frame.shape[2]

        # cropped_face = cv2.resize(temp_frame, (512, 512), interpolation=cv2.INTER_LINEAR)
        cropped_face = F.interpolate(temp_frame, size=512, mode='bilinear', align_corners=False)

        result = self.model_gpen.process(cropped_face)
        scale_factor = int(result.shape[1] / input_size)
        return result, scale_factor

    def Release(self):
        self.model_gpen = None
