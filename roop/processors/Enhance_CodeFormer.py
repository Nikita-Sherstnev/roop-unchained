from typing import Any, List, Callable
import cv2
import threading
import numpy as np
import torch
import roop.globals

from roop.typing import Face, Frame, FaceSet
from roop.utilities import resolve_relative_path
from roop.models.codeformer import CodeFormer, img2tensor, tensor2img
from torchvision.transforms.functional import normalize as normalize_torch


# THREAD_LOCK = threading.Lock()


class Enhance_CodeFormer():
    model = None
    devicename = None

    processorname = 'codeformer'
    type = 'enhance'


    def Initialize(self, devicename: str):
        if self.model is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_path = resolve_relative_path('../models/codeformer.pth')

            self.model = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                            connect_list=['32', '64', '128', '256']).to(self.devicename)

            checkpoint = torch.load(model_path)['params_ema']
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()

    def Run(self, source_faceset: FaceSet, target_face: Face, temp_frame: Frame) -> Frame:
        input_size = temp_frame.shape[1]

        cropped_face = cv2.resize(temp_frame, (512, 512), interpolation=cv2.INTER_LINEAR)
        cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
        normalize_torch(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

        fidelity_weight = 0.5
        with torch.no_grad():
            output = self.model(cropped_face_t, w=fidelity_weight, adain=True)[0]
            result = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))

        scale_factor = int(result.shape[1] / input_size)
        return result.astype(np.uint8), scale_factor

    def Release(self):
        del self.model
        self.model = None
