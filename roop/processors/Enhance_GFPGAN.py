from typing import Any, List, Callable
import cv2
import numpy as np
import roop.globals
import gfpgan

from roop.typing import Face, Frame, FaceSet
from roop.utilities import resolve_relative_path


# THREAD_LOCK = threading.Lock()


class Enhance_GFPGAN():

    model_gfpgan = None
    name = None
    devicename = None

    processorname = 'gfpgan'
    type = 'enhance'


    def Initialize(self, devicename):
        if self.model_gfpgan is None:
            model_path = resolve_relative_path('../models/GFPGANv1.4.pth')
            self.model_gfpgan = gfpgan.GFPGANer(model_path=model_path, upscale=4)
            # replace Mac mps with cpu for the moment
            devicename = devicename.replace('mps', 'cpu')
            self.devicename = devicename

        self.name = 'input'

    def Run(self, source_faceset: FaceSet, target_face: Face, temp_frame: Frame) -> Frame:
        input_size = temp_frame.shape[1]

        _, _, result = self.model_gfpgan.enhance(
            temp_frame,
            paste_back=True
        )

        scale_factor = int(result.shape[1] / input_size)
        return result.astype(np.uint8), scale_factor


    def Release(self):
        self.model_gfpgan = None











