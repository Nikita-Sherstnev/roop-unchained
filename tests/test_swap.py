import os
import pathlib

import pytest
import numpy as np

import cv2
from settings import Settings
from roop.face_util import extract_face_images
from roop.core import batch_process
import roop.globals
from roop.ProcessEntry import ProcessEntry
from roop.FaceSet import FaceSet
from roop.ProcessMgr import ProcessMgr
from roop.ProcessOptions import ProcessOptions
from roop.processors.FaceSwapInsightFace import FaceSwapInsightFace


def mock_progress_gradio(*args, **kwargs):
    pass


class MockInsightFace:
    processorname = 'faceswap'
    type = 'swap'

    def __init__(self, fake_frame):
        self.fake_frame = fake_frame

    def Run(self, *args):
        return self.fake_frame


class TestSwapFace:
    process_mgr = ProcessMgr(None)

    @pytest.fixture(autouse=True)
    def config(self):
        roop.globals.CFG = Settings('config.yaml')
        roop.globals.execution_threads = roop.globals.CFG.max_threads
        roop.globals.video_encoder = roop.globals.CFG.output_video_codec
        roop.globals.video_quality = roop.globals.CFG.video_quality
        roop.globals.max_memory = roop.globals.CFG.memory_limit if roop.globals.CFG.memory_limit > 0 else None
        roop.globals.CFG.force_cpu = False
        roop.globals.output_path = 'output'
        roop.globals.face_swap_mode = 'first'
        # roop.globals.execution_providers = ['CUDAExecutionProvider']

    @pytest.fixture(scope="function", autouse=True)
    def clear_inputs_targets(self):
        roop.globals.INPUT_FACESETS = []
        roop.globals.TARGET_FACES = []
        roop.globals.frame_processors = []

    def test_swap_one_face(self):
        source = 'tests/assets/ana-de-armas.jpg'
        target = 'tests/assets/wonder-woman.jpg'

        source_face = extract_face_images(source, (False, 0))[0][0]
        source_face.mask_offsets = (0,0)

        face_set = FaceSet()
        face_set.faces.append(source_face)
        roop.globals.INPUT_FACESETS.append(face_set)

        target_face = extract_face_images(target, (False, 0))[0][0]

        roop.globals.TARGET_FACES.append(target_face)

        use_clip = False
        clip_text = None
        processing_method = "In-Memory processing"

        list_files_process: list[ProcessEntry] = []
        list_files_process.append(ProcessEntry(target, 0,0, 0))

        batch_process(list_files_process, use_clip, clip_text,
                      processing_method == "In-Memory processing", mock_progress_gradio)

        outdir = pathlib.Path(roop.globals.output_path)
        result = [item for item in outdir.rglob(f"*{target.rsplit('/', maxsplit=1)[-1].split('.')[0]}*") \
                  if item.is_file()][0]

        swapped_face = extract_face_images(str(result), (False, 0))

        source_similarity = np.dot(source_face['embedding'], swapped_face[0][0]['embedding'])
        target_similarity = np.dot(target_face['embedding'], swapped_face[0][0]['embedding'])

        assert source_similarity > target_similarity * 10
        os.remove(result)

    @pytest.mark.parametrize('enhancement', ['gfpgan', 'gpen', 'codeformer', 'dmdnet', 'restoreformer'])
    def test_enhancement(self, enhancement):
        source = 'tests/assets/ana-de-armas.jpg'
        target = 'tests/assets/wonder-woman.jpg'
        fake_frame = 'tests/assets/fake-frame.jpg'

        enhanced_path = 'tests/enhanced.jpg'

        source_face = extract_face_images(source, (False, 0))[0][0]
        source_face.mask_offsets = (0,0)

        face_set = FaceSet()
        face_set.faces.append(source_face)

        target_face = extract_face_images(target, (False, 0))[0][0]
        target_face.matrix = np.array([[ 4.28439966e-01, -2.93581036e-02, -2.89326694e+01],
                                      [ 2.93581036e-02,  4.28439966e-01, -3.45334194e+01]])

        fake_frame = cv2.imread(fake_frame)
        target_img = cv2.imread(target)

        options = ProcessOptions(enhancement, roop.globals.distance_threshold,
                                 roop.globals.blend_ratio, 'first', 0, None)
        self.process_mgr.initialize([face_set], [target_face], options)

        self.process_mgr.processors.insert(0, MockInsightFace(fake_frame))

        newframe = self.process_mgr.process_face(0, target_face, target_img)

        cv2.imwrite(enhanced_path, newframe)

        file_size1 = os.path.getsize('tests/assets/ww-swap.jpg')
        file_size2 = os.path.getsize(enhanced_path)

        assert file_size1 < file_size2
        self.process_mgr.processors.pop(0)
        os.remove(enhanced_path)

    def test_swap_gif(self):
        source = 'tests/assets/ana-de-armas.jpg'
        target = 'tests/assets/wonder-woman.gif'

        source_face = extract_face_images(source, (False, 0))[0][0]
        source_face.mask_offsets = (0,0)

        face_set = FaceSet()
        face_set.faces.append(source_face)
        roop.globals.INPUT_FACESETS.append(face_set)

        target_face = extract_face_images(target, (True, 0))[0][0]

        roop.globals.TARGET_FACES.append(target_face)

        use_clip = False
        clip_text = None
        processing_method = "In-Memory processing"

        list_files_process: list[ProcessEntry] = []
        list_files_process.append(ProcessEntry(target, 0,0, 0))

        batch_process(list_files_process, use_clip, clip_text,
                      processing_method == "In-Memory processing", mock_progress_gradio)

        outdir = pathlib.Path(roop.globals.output_path)
        result = [item for item in outdir.rglob(f"*{target.rsplit('/', maxsplit=1)[-1].split('.')[0]}*.gif") \
                  if item.is_file()][0]

        swapped_face = extract_face_images(str(result), (True, 0))

        source_similarity = np.dot(source_face['embedding'], swapped_face[0][0]['embedding'])
        target_similarity = np.dot(target_face['embedding'], swapped_face[0][0]['embedding'])

        assert source_similarity > target_similarity * 10
        os.remove(result)

    def test_clip_to_seg(self):
        source = 'tests/assets/ana-de-armas.jpg'
        target = 'tests/assets/wonder-woman.jpg'

        source_face = extract_face_images(source, (False, 0))[0][0]
        source_face.mask_offsets = (0,0)

        face_set = FaceSet()
        face_set.faces.append(source_face)
        roop.globals.INPUT_FACESETS.append(face_set)

        target_face = extract_face_images(target, (False, 0))[0][0]

        roop.globals.TARGET_FACES.append(target_face)

        use_clip = True
        clip_text = 'hair'
        processing_method = "In-Memory processing"

        list_files_process: list[ProcessEntry] = []
        list_files_process.append(ProcessEntry(target, 0,0, 0))

        batch_process(list_files_process, use_clip, clip_text,
                      processing_method == "In-Memory processing", mock_progress_gradio)

        outdir = pathlib.Path(roop.globals.output_path)
        result = [item for item in outdir.rglob(f"*{target.rsplit('/', maxsplit=1)[-1].split('.')[0]}*") \
                  if item.is_file()][0]

        swapped_face = extract_face_images(str(result), (False, 0))

        source_similarity = np.dot(source_face['embedding'], swapped_face[0][0]['embedding'])
        target_similarity = np.dot(target_face['embedding'], swapped_face[0][0]['embedding'])

        assert source_similarity > target_similarity * 10
        os.remove(result)