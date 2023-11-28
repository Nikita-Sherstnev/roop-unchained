from typing import Any, List, Callable
import cv2
import threading
import numpy as np
import roop.globals

from roop.typing import Face, Frame, FaceSet
from roop.utilities import resolve_relative_path
from roop.models.restoreformer import RestoreFormer


class Enhance_Restoreformer():
    model_restoreformer = None
    name = None
    devicename = None

    processorname = 'restoreformer'
    type = 'enhance'


    def Initialize(self, devicename: str):
        model_path = resolve_relative_path('../models/Restoreformer/RestoreFormer.ckpt')

        self.model_restoreformer = RestoreFormer(
                        model_path=model_path,
                        upscale=2,
                        arch='RestoreFormer',
                        bg_upsampler=None)


    def Run(self, source_faceset: FaceSet, target_face: Face, temp_frame: Frame) -> Frame:
        input_size = temp_frame.shape[1]

        _, restored_faces, restored_img = self.model_restoreformer.enhance(
                                                            temp_frame,
                                                            has_aligned=True,
                                                            only_center_face=True,
                                                            paste_back=False)

        result = restored_faces[0]
        scale_factor = int(result.shape[1] / input_size)
        return result, scale_factor


    def Release(self):
        del self.model_restoreformer
        self.model_restoreformer = None
