import torch
import numpy as np
from torchvision import transforms
from roop.processors.frame import face_align


class Attribute:
    def __init__(self, model_file=None, device=None):
        assert model_file is not None
        self.model_file = model_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        self.model = torch.load(model_file)
        self.model.to(self.device)
        self.model.eval()

        self.input_mean = 127.5
        self.input_std = 128.0

        self.input_size = (96, 96)
        self.input_shape = (1, 3, 96, 96)

        self.taskname = 'genderage'

    def prepare(self, ctx_id, **kwargs):
        if ctx_id < 0:
            self.device = 'cpu'
            self.model.to(self.device)

    def get(self, img, face):
        bbox = face.bbox
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        _scale = self.input_size[0] / (max(w, h) * 1.5)

        aimg, M = face_align.transform(img, center, self.input_size[0], _scale, rotate)
        input_size = tuple(aimg.shape[0:2][::-1])

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
        ])

        aimg = transform(aimg)

        img = (aimg - self.input_mean) / self.input_std
        blob = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(blob)[0].cpu()

        assert len(pred)==3
        gender = np.argmax(pred[:2])
        age = int(np.round(pred[2]*100))
        face['gender'] = gender.numpy()
        face['age'] = age
        return gender, age
