from typing import Any, List, Callable
import cv2
import numpy as np
import onnxruntime
import roop.globals

from roop.typing import Face, Frame, FaceSet
from roop.utilities import resolve_relative_path


class Enhance_GPEN():

    model_gpen = None
    name = None
    devicename = None

    processorname = 'gpen'
    type = 'enhance'


    def Initialize(self, devicename):
        if self.model_gpen is None:
            model_path = resolve_relative_path('../models/GPEN-BFR-512.onnx')
            self.model_gpen = onnxruntime.InferenceSession(model_path, None, providers=roop.globals.execution_providers)
            # replace Mac mps with cpu for the moment
            devicename = devicename.replace('mps', 'cpu')
            self.devicename = devicename

        self.name = self.model_gpen.get_inputs()[0].name

    def Run(self, source_faceset: FaceSet, target_face: Face, temp_frame: Frame) -> Frame:
        # preprocess
        input_size = temp_frame.shape[1]
        temp_frame = cv2.resize(temp_frame, (512, 512), cv2.INTER_CUBIC)

        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        temp_frame = temp_frame.astype('float32') / 255.0
        temp_frame = (temp_frame - 0.5) / 0.5
        temp_frame = np.expand_dims(temp_frame, axis=0).transpose(0, 3, 1, 2)

        io_binding = self.model_gpen.io_binding()
        io_binding.bind_cpu_input("input", temp_frame)
        io_binding.bind_output("output", self.devicename)
        self.model_gpen.run_with_iobinding(io_binding)
        ort_outs = io_binding.copy_outputs_to_cpu()
        result = ort_outs[0][0]

        # post-process
        result = np.clip(result, -1, 1)
        result = (result + 1) / 2
        result = result.transpose(1, 2, 0) * 255.0
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        scale_factor = int(result.shape[1] / input_size)
        return result.astype(np.uint8), scale_factor


    def Release(self):
        self.model_gpen = None
