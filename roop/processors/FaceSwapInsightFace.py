from typing import Any, List, Callable
import roop.globals
import cv2
import numpy as np

from roop.typing import Face, Frame
from roop.utilities import resolve_relative_path
from roop.models.inswapper import INSwapper



class FaceSwapInsightFace():
    model_swap_insightface = None


    processorname = 'faceswap'
    type = 'swap'


    def Initialize(self, devicename):
        if self.model_swap_insightface is None:
            model_path = resolve_relative_path('../models/inswapper_128.pt')
            self.model_swap_insightface = INSwapper(model_path)


    def Run(self, source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
        img_fake, M = self.model_swap_insightface.get(temp_frame, target_face, source_face, paste_back=False)
        target_face.matrix = M
        return img_fake


    def Release(self):
        del self.model_swap_insightface
        self.model_swap_insightface = None
