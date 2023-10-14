import os
import os.path as osp
from pathlib import Path
import pickle

import torch
import numpy as np
import cv2
from roop.processors.frame import face_align, transform


def get_object(name):
    objects_dir = 'models'
    if not name.endswith('.pkl'):
        name = name+".pkl"
    filepath = osp.join(objects_dir, name)
    if not osp.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    print(obj.shape)
    return obj


class Landmark:
    def __init__(self, model_file=None, device=None):
        assert model_file is not None
        self.model_file = model_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        self.model = torch.load(model_file)
        self.model.to(self.device)
        self.model.eval()

        for layer in self.model.children():
            if hasattr(layer, 'out_features'):
                output_shape = layer.out_features

        self.input_mean = 0.0
        self.input_std = 1.0

        input_shape = ['None', 3, 192, 192]
        self.input_size = tuple(input_shape[2:4][::-1])
        self.input_shape = input_shape

        self.require_pose = False

        if output_shape==3309:
            self.lmk_dim = 3
            self.lmk_num = 68
            self.mean_lmk = get_object('meanshape_68.pkl')
            self.require_pose = True
        else:
            self.lmk_dim = 2
            self.lmk_num = output_shape//self.lmk_dim
        self.taskname = 'landmark_%dd_%d'%(self.lmk_dim, self.lmk_num)

    def prepare(self, ctx_id, **kwargs):
        pass

    def get(self, img, face):
        bbox = face.bbox
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        _scale = self.input_size[0]  / (max(w, h)*1.5)

        aimg, M = face_align.transform(img, center, self.input_size[0], _scale, rotate)
        input_size = tuple(aimg.shape[0:2][::-1])

        blob = cv2.dnn.blobFromImage(aimg, 1.0/self.input_std, input_size,
                                    (self.input_mean, self.input_mean, self.input_mean),
                                    swapRB=True)
        blob = torch.tensor(blob).to(self.device)
        with torch.inference_mode():
            pred = self.model(blob)[0].cpu().numpy()
        if pred.shape[0] >= 3000:
            pred = pred.reshape((-1, 3))
        else:
            pred = pred.reshape((-1, 2))
        if self.lmk_num < pred.shape[0]:
            pred = pred[self.lmk_num*-1:,:]
        pred[:, 0:2] += 1
        pred[:, 0:2] *= (self.input_size[0] // 2)
        if pred.shape[1] == 3:
            pred[:, 2] *= (self.input_size[0] // 2)

        IM = cv2.invertAffineTransform(M)
        pred = face_align.trans_points(pred, IM)
        face[self.taskname] = pred
        if self.require_pose:
            P = transform.estimate_affine_matrix_3d23d(self.mean_lmk, pred)
            s, R, t = transform.P2sRt(P)
            rx, ry, rz = transform.matrix2angle(R)
            pose = np.array( [rx, ry, rz], dtype=np.float32 )
            face['pose'] = pose #pitch, yaw, roll
        return pred