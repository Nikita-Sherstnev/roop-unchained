import torch
import numpy as np
from torchvision import transforms

from roop.processors.frame import face_align


class ArcFace:
    def __init__(self, model_file=None, device='cuda'):
        assert model_file is not None
        self.model_file = model_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        self.model = torch.load(model_file)
        self.model.to(self.device)
        self.model.eval()

        self.input_mean = 127.5
        self.input_std = 127.5

        self.input_size = (112, 112)

    def prepare(self, ctx_id, **kwargs):
        pass

    def get(self, img, face):
        aimg = face_align.norm_crop(img, landmark=face.kps, image_size=self.input_size[0])
        face.embedding = self.get_feat(aimg).flatten()
        return face.embedding

    def compute_sim(self, feat1, feat2):
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
        return sim

    def get_feat(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        img_tensors = [transform(img) for img in imgs]
        batch = torch.stack(img_tensors).to(self.device)

        with torch.no_grad():
            net_out = self.model(batch)

        return net_out.cpu()

    def forward(self, batch_data):
        blob = (batch_data - self.input_mean) / self.input_std
        blob = torch.tensor(blob).to(self.device)
        net_out = self.model(blob)
        return net_out.cpu()
