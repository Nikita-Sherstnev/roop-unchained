import torch

from roop.processors.frame import face_align


class INSwapper():
    def __init__(self, model_file=None, device=None):
        self.model_file = model_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        self.model = torch.load(model_file)
        self.model.to(self.device)
        self.model.eval()

        self.input_mean = 0.0
        self.input_std = 255.0
        self.emap = self.model.state_dict()['emap']

        self.input_names = ['img', 'latent']
        self.output_names = ['output']

        input_shape = [1, 3, 128, 128]
        self.input_shape = input_shape
        print('inswapper-shape:', self.input_shape)

        self.input_size = tuple(input_shape[2:4][::-1])

        self.model.eval()

    def get(self, img, target_face, source_face):
        aimg, M = face_align.norm_crop2(img, target_face.kps, self.input_size[0])

        # Swap R and B channels
        blob = torch.from_numpy(aimg / 255.0)[:,:,[2, 1, 0]].permute(2,0,1) \
                    .unsqueeze(0).float().to(self.device)

        latent = source_face.normed_embedding.reshape((1,-1)).to(self.device)
        latent = torch.mm(latent, self.emap) / torch.norm(latent)

        with torch.inference_mode():
            pred = self.model(blob, latent)

            return pred, M
