import os

import cv2
import torch
from .codeformer import img2tensor, tensor2img
from torchvision.transforms.functional import normalize

from roop.models.vqvae_arch import VQVAEGANMultiHeadTransformer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from urllib.parse import urlparse

from torch.hub import download_url_to_file, get_dir


def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Load file form http url, will download models if necessary.

    Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py

    Args:
        url (str): URL to be downloaded.
        model_dir (str): The path to save the downloaded model. Should be a full path. If None, use pytorch hub_dir.
            Default: None.
        progress (bool): Whether to show the download progress. Default: True.
        file_name (str): The downloaded file name. If None, use the file name in the url. Default: None.

    Returns:
        str: The path to the downloaded file.
    """
    if model_dir is None:  # use the pytorch hub_dir
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file


class RestoreFormer():
    """Helper for restoration with RestoreFormer.

    It will detect and crop faces, and then resize the faces to 512x512.
    RestoreFormer is used to restored the resized faces.
    The background is upsampled with the bg_upsampler.
    Finally, the faces will be pasted back to the upsample background image.

    Args:
        model_path (str): The path to the GFPGAN model. It can be urls (will first download it automatically).
        upscale (float): The upscale of the final output. Default: 2.
        arch (str): The RestoreFormer architecture. Option: RestoreFormer | RestoreFormer++. Default: RestoreFormer++.
        bg_upsampler (nn.Module): The upsampler for the background. Default: None.
    """

    def __init__(self, model_path, upscale=2, arch='RestoreFromerPlusPlus', bg_upsampler=None, device=None):
        self.upscale = upscale
        self.bg_upsampler = bg_upsampler
        self.arch = arch

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        if arch == 'RestoreFormer':
            self.RF = VQVAEGANMultiHeadTransformer(head_size = 8, ex_multi_scale_num = 0)
        elif arch == 'RestoreFormer++':
            self.RF = VQVAEGANMultiHeadTransformer(head_size = 4, ex_multi_scale_num = 1)
        else:
            raise NotImplementedError(f'Not support arch: {arch}.')


        if model_path.startswith('https://'):
            model_path = load_file_from_url(
                url=model_path, model_dir=os.path.join(ROOT_DIR, 'experiments/weights'), progress=True, file_name=None)
        loadnet = torch.load(model_path)

        strict=False
        weights = loadnet['state_dict']
        new_weights = {}
        for k, v in weights.items():
            if k.startswith('vqvae.'):
                k = k.replace('vqvae.', '')
            new_weights[k] = v
        self.RF.load_state_dict(new_weights, strict=strict)

        self.RF.eval()
        self.RF = self.RF.to(self.device)

    @torch.no_grad()
    def enhance(self, img, has_aligned=False, only_center_face=False, paste_back=True):
        if has_aligned:  # the inputs are already aligned
            img = cv2.resize(img, (512, 512))
            cropped_faces = [img]
        # else:
        #     self.face_helper.read_image(img)
        #     self.face_helper.get_face_landmarks_5(only_center_face=only_center_face, eye_dist_threshold=5)
        #     # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
        #     # TODO: even with eye_dist_threshold, it will still introduce wrong detections and restorations.
        #     # align and warp each face
        #     self.face_helper.align_warp_face()

        restored_faces = []
        # face restoration
        for cropped_face in cropped_faces:
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                output = self.RF(cropped_face_t)[0]
                restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
            except RuntimeError as error:
                print(f'\tFailed inference for RestoreFormer: {error}.')
                restored_face = cropped_face

            restored_face = restored_face.astype('uint8')
            restored_faces.append(restored_face)

        return cropped_faces, restored_faces, None
