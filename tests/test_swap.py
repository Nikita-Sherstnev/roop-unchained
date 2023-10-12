import pytest
import pathlib

from settings import Settings
from roop.face_util import extract_face_images
from roop.core import batch_process
import roop.globals
from roop.ProcessEntry import ProcessEntry
from roop.FaceSet import FaceSet


def progress_gradio(*args, **kwargs):
    pass


class TestSwapFace:
    @pytest.fixture(autouse=True)
    def config(self):
        roop.globals.CFG = Settings('config.yaml')
        roop.globals.execution_threads = roop.globals.CFG.max_threads
        roop.globals.video_encoder = roop.globals.CFG.output_video_codec
        roop.globals.video_quality = roop.globals.CFG.video_quality
        roop.globals.max_memory = roop.globals.CFG.memory_limit if roop.globals.CFG.memory_limit > 0 else None
        roop.globals.CFG.force_cpu = True
        roop.globals.output_path = 'output'
        roop.globals.face_swap_mode = 'first'

    def test_swap(self):
        source = 'tests/assets/ana-de-armas.jpg'
        target = 'tests/assets/wonder-woman.jpg'

        SELECTION_FACES_DATA = extract_face_images(source, (False, 0))

        face = SELECTION_FACES_DATA[0][0]
        face.mask_offsets = (0,0)
        face_set = FaceSet()
        face_set.faces.append(face)
        roop.globals.INPUT_FACESETS.append(face_set)

        SELECTION_FACES_DATA = extract_face_images(target, (False, 0))

        roop.globals.TARGET_FACES.append(SELECTION_FACES_DATA[0][0])

        use_clip = False
        clip_text = None
        processing_method = "In-Memory processing"

        list_files_process: list[ProcessEntry] = []
        list_files_process.append(ProcessEntry(target, 0,0, 0))

        batch_process(list_files_process, use_clip, clip_text,
                      processing_method == "In-Memory processing", progress_gradio)
        is_processing = False
        outdir = pathlib.Path(roop.globals.output_path)
        outfiles = [item for item in outdir.rglob(f"{target.split('/')[-1].split('.')[0]}*") if item.is_file()]
