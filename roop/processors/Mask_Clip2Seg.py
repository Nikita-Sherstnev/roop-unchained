import os
import cv2
import numpy as np
import torch
import threading
from torchvision import transforms
from clip.clipseg import CLIPDensePredT
from numpy import asarray
from typing import Any, List, Callable
import numpy as np

from roop.typing import Face, Frame
from roop.utilities import resolve_relative_path

THREAD_LOCK_CLIP = threading.Lock()


class Mask_Clip2Seg():

    model_clip = None

    processorname = 'clip2seg'
    type = 'mask'


    def Initialize(self, devicename):
        if self.model_clip is None:
            self.model_clip = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
            self.model_clip.eval();
            self.model_clip.load_state_dict(torch.load('models/CLIP/rd64-uni-refined.pth', map_location=torch.device('cpu')), strict=False)

        device = torch.device(devicename)
        self.model_clip.to(device)


    def Run(self, img1, keywords:str) -> Frame:
        if keywords is None or len(keywords) < 1 or img1 is None:
            return img1

        source_image_small = cv2.resize(img1, (256,256))

        img_mask = np.full((source_image_small.shape[0],source_image_small.shape[1]), 0, dtype=np.float32)
        mask_border = 1
        l = 0
        t = 0
        r = 1
        b = 1

        mask_blur = 5
        clip_blur = 5

        img_mask = cv2.rectangle(img_mask, (mask_border+int(l), mask_border+int(t)),
                                (256 - mask_border-int(r), 256-mask_border-int(b)), (255, 255, 255), -1)
        img_mask = cv2.GaussianBlur(img_mask, (mask_blur*2+1,mask_blur*2+1), 0)
        img_mask /= 255


        input_image = source_image_small

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((256, 256)),
        ])
        img = transform(input_image).unsqueeze(0)

        thresh = 0.5
        prompts = keywords.split(',')
        with THREAD_LOCK_CLIP:
            with torch.no_grad():
                preds = self.model_clip(img.repeat(len(prompts),1,1,1), prompts)[0]
        clip_mask = torch.sigmoid(preds[0][0])
        for i in range(len(prompts)-1):
            clip_mask += torch.sigmoid(preds[i+1][0])

        clip_mask = clip_mask.data.cpu().numpy()
        np.clip(clip_mask, 0, 1)

        clip_mask[clip_mask>thresh] = 1.0
        clip_mask[clip_mask<=thresh] = 0.0
        kernel = np.ones((5, 5), np.float32)
        clip_mask = cv2.dilate(clip_mask, kernel, iterations=1)
        clip_mask = cv2.GaussianBlur(clip_mask, (clip_blur*2+1,clip_blur*2+1), 0)

        img_mask *= clip_mask
        img_mask[img_mask<0.0] = 0.0
        return img_mask



    def Release(self):
        self.model_clip = None

