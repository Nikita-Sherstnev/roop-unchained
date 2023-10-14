import os
import shutil
import pathlib
import gradio as gr
import roop.utilities as util
import roop.globals
import ui.globals
from roop.face_util import extract_face_images
from roop.capturer import get_video_frame, get_video_frame_total, get_image_frame
from roop.ProcessEntry import ProcessEntry
from roop.FaceSet import FaceSet

last_image = None


IS_INPUT = True
SELECTED_FACE_INDEX = 0

SELECTED_INPUT_FACE_INDEX = 0
SELECTED_TARGET_FACE_INDEX = 0

input_faces = None
target_faces = None
face_selection = None

selected_preview_index = 0

is_processing = False            

list_files_process : list[ProcessEntry] = []
no_face_choices = ["Use untouched original frame","Retry rotated", "Skip Frame"]


def faceswap_tab():
    global no_face_choices

    with gr.Tab("🎭 Face Swap"):
        with gr.Row(variant='panel'):
            with gr.Column(scale=2):
                with gr.Row():
                    with gr.Column(min_width=160):
                        input_faces = gr.Gallery(label="Input faces", allow_preview=True, preview=True, height=128, object_fit="scale-down")
                        with gr.Accordion(label="Advanced Settings", open=False):
                            mask_top = gr.Slider(0, 256, value=0, label="Offset Face Top", step=1.0, interactive=True)
                            mask_bottom = gr.Slider(0, 256, value=0, label="Offset Face Bottom", step=1.0, interactive=True)
                        bt_remove_selected_input_face = gr.Button("❌ Remove selected", size='sm')
                        bt_clear_input_faces = gr.Button("💥 Clear all", variant='stop', size='sm')
                    with gr.Column(min_width=160):
                        target_faces = gr.Gallery(label="Target faces", allow_preview=True, preview=True, height=128, object_fit="scale-down")
                        bt_remove_selected_target_face = gr.Button("❌ Remove selected", size='sm')
                        bt_add_local = gr.Button('Add local files from', size='sm')
                        local_folder = gr.Textbox(show_label=False, placeholder="/content/", interactive=True)
                with gr.Row(variant='panel'):
                    bt_srcfiles = gr.Files(label='Source File(s)', file_count="multiple", file_types=["image", ".fsz"], elem_id='filelist', height=233)
                    bt_destfiles = gr.Files(label='Target File(s)', file_count="multiple", file_types=["image", "video"], elem_id='filelist', height=233)
                with gr.Row(variant='panel'):
                    gr.Markdown('')
                    forced_fps = gr.Slider(minimum=0, maximum=120, value=0, label="Video FPS", info='Overrides detected fps if not 0', step=1.0, interactive=True, container=True)

            with gr.Column(scale=2):
                previewimage = gr.Image(label="Preview Image", height=576, interactive=False)
                with gr.Row(variant='panel'):
                        fake_preview = gr.Checkbox(label="Face swap frames", value=False)
                        bt_refresh_preview = gr.Button("🔄 Refresh", variant='secondary', size='sm')
                        bt_use_face_from_preview = gr.Button("Use Face from this Frame", variant='primary', size='sm')
                with gr.Row():
                    preview_frame_num = gr.Slider(0, 0, value=0, label="Frame Number", step=1.0, interactive=True)
                with gr.Row():
                    text_frame_clip = gr.Markdown('Processing frame range [0 - 0]')
                    set_frame_start = gr.Button("⬅ Set as Start", size='sm')
                    set_frame_end = gr.Button("➡ Set as End", size='sm')
        with gr.Row(visible=False) as dynamic_face_selection:
            with gr.Column(scale=2):
                face_selection = gr.Gallery(label="Detected faces", allow_preview=True, preview=True, height=256, object_fit="scale-down")
            with gr.Column():
                bt_faceselect = gr.Button("☑ Use selected face", size='sm')
                bt_cancelfaceselect = gr.Button("Done", size='sm')
            with gr.Column():
                gr.Markdown(' ') 
    
        with gr.Row(variant='panel'):
            with gr.Column(scale=1):
                selected_face_detection = gr.Dropdown(["First found", "All faces", "Selected face", "All female", "All male"], value="First found", label="Select face selection for swapping")
                max_face_distance = gr.Slider(0.01, 1.0, value=0.65, label="Max Face Similarity Threshold")
                video_swapping_method = gr.Dropdown(["Extract Frames to media","In-Memory processing"], value="In-Memory processing", label="Select video processing method", interactive=True)
                no_face_action = gr.Dropdown(choices=no_face_choices, value=no_face_choices[0], label="Action on no face detected", interactive=True)
            with gr.Column(scale=1):
                ui.globals.ui_selected_enhancer = gr.Dropdown(["None", "Codeformer", "DMDNet", "GFPGAN", "GPEN"], value="None", label="Select post-processing")
                ui.globals.ui_blend_ratio = gr.Slider(0.0, 1.0, value=0.65, label="Original/Enhanced image blend ratio")
                with gr.Box():
                    roop.globals.skip_audio = gr.Checkbox(label="Skip audio", value=False)
                    roop.globals.keep_frames = gr.Checkbox(label="Keep Frames (relevant only when extracting frames)", value=False)
                    roop.globals.wait_after_extraction = gr.Checkbox(label="Wait for user key press before creating video ", value=False)
            with gr.Column(scale=1):
                chk_useclip = gr.Checkbox(label="Use Text Masking", value=False)
                clip_text = gr.Textbox(label="List of objects to mask and restore back on fake image", placeholder="cup,hands,hair,banana" ,elem_id='tooltip')
                gr.Dropdown(["Clip2Seg"], value="Clip2Seg", label="Engine")
                bt_preview_mask = gr.Button("👥 Show Mask Preview", variant='secondary')
                    
        with gr.Row(variant='panel'):
            with gr.Column():
                bt_start = gr.Button("▶ Start", variant='primary')
                gr.Button("👀 Open Output Folder", size='sm').click(fn=lambda: util.open_folder(roop.globals.output_path))
            with gr.Column():
                bt_stop = gr.Button("⏹ Stop", variant='secondary')
            with gr.Column(scale=2):
                gr.Markdown(' ') 
        with gr.Row(variant='panel'):
            with gr.Column():
                resultfiles = gr.Files(label='Processed File(s)', interactive=False)
            with gr.Column():
                resultimage = gr.Image(type='filepath', label='Final Image', interactive=False )
                resultvideo = gr.Video(label='Final Video', interactive=False, visible=False)

    previewinputs = [preview_frame_num, bt_destfiles, fake_preview, ui.globals.ui_selected_enhancer, selected_face_detection,
                        max_face_distance, ui.globals.ui_blend_ratio, chk_useclip, clip_text, no_face_action] 
    input_faces.select(on_select_input_face, None, None).then(fn=on_preview_frame_changed, inputs=previewinputs, outputs=[previewimage, mask_top, mask_bottom])
    bt_remove_selected_input_face.click(fn=remove_selected_input_face, outputs=[input_faces])
    bt_srcfiles.change(fn=on_srcfile_changed, show_progress='full', inputs=bt_srcfiles, outputs=[dynamic_face_selection, face_selection, input_faces])

    mask_top.input(fn=on_mask_top_changed, inputs=[mask_top], show_progress='hidden')
    mask_bottom.input(fn=on_mask_bottom_changed, inputs=[mask_bottom], show_progress='hidden')


    target_faces.select(on_select_target_face, None, None)
    bt_remove_selected_target_face.click(fn=remove_selected_target_face, outputs=[target_faces])

    forced_fps.change(fn=on_fps_changed, inputs=[forced_fps], show_progress='hidden')
    bt_destfiles.change(fn=on_destfiles_changed, inputs=[bt_destfiles], outputs=[preview_frame_num, text_frame_clip], show_progress='hidden').then(fn=on_preview_frame_changed, inputs=previewinputs, outputs=[previewimage, mask_top, mask_bottom], show_progress='full')
    bt_destfiles.select(fn=on_destfiles_selected, outputs=[preview_frame_num, text_frame_clip, forced_fps], show_progress='hidden').then(fn=on_preview_frame_changed, inputs=previewinputs, outputs=[previewimage, mask_top, mask_bottom], show_progress='hidden')
    bt_destfiles.clear(fn=on_clear_destfiles, outputs=[target_faces])
    resultfiles.select(fn=on_resultfiles_selected, inputs=[resultfiles], outputs=[resultimage, resultvideo])

    face_selection.select(on_select_face, None, None)
    bt_faceselect.click(fn=on_selected_face, outputs=[input_faces, target_faces, selected_face_detection])
    bt_cancelfaceselect.click(fn=on_end_face_selection, outputs=[dynamic_face_selection, face_selection])
    
    bt_clear_input_faces.click(fn=on_clear_input_faces, outputs=[input_faces])


    bt_add_local.click(fn=on_add_local_folder, inputs=[local_folder], outputs=[bt_destfiles])
    bt_preview_mask.click(fn=on_preview_mask, inputs=[preview_frame_num, bt_destfiles, clip_text], outputs=[previewimage]) 

    start_event = bt_start.click(fn=start_swap, 
        inputs=[ui.globals.ui_selected_enhancer, selected_face_detection, roop.globals.keep_frames,
                    roop.globals.skip_audio, max_face_distance, ui.globals.ui_blend_ratio, chk_useclip, clip_text,video_swapping_method, no_face_action],
        outputs=[bt_start, resultfiles])
    after_swap_event = start_event.then(fn=on_resultfiles_finished, inputs=[resultfiles], outputs=[resultimage, resultvideo])
    
    bt_stop.click(fn=stop_swap, cancels=[start_event, after_swap_event], queue=False)
    
    bt_refresh_preview.click(fn=on_preview_frame_changed, inputs=previewinputs, outputs=[previewimage, mask_top, mask_bottom])            
    fake_preview.change(fn=on_preview_frame_changed, inputs=previewinputs, outputs=[previewimage, mask_top, mask_bottom])
    preview_frame_num.change(fn=on_preview_frame_changed, inputs=previewinputs, outputs=[previewimage, mask_top, mask_bottom], show_progress='hidden')
    bt_use_face_from_preview.click(fn=on_use_face_from_selected, show_progress='full', inputs=[bt_destfiles, preview_frame_num], outputs=[dynamic_face_selection, face_selection, target_faces, selected_face_detection])
    set_frame_start.click(fn=on_set_frame, inputs=[set_frame_start, preview_frame_num], outputs=[text_frame_clip])
    set_frame_end.click(fn=on_set_frame, inputs=[set_frame_end, preview_frame_num], outputs=[text_frame_clip])

                     
            
def on_mask_top_changed(mask_offset):
    global SELECTED_INPUT_FACE_INDEX

    if len(roop.globals.INPUT_FACESETS) > SELECTED_INPUT_FACE_INDEX:
        roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0].mask_offsets[0] = mask_offset

def on_mask_bottom_changed(mask_offset):
    global SELECTED_INPUT_FACE_INDEX

    if len(roop.globals.INPUT_FACESETS) > SELECTED_INPUT_FACE_INDEX:
        roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0].mask_offsets[1] = mask_offset




def on_add_local_folder(folder):
    files = util.get_local_files_from_folder(folder)
    if files is None:
        gr.Warning("Empty folder or folder not found!")
    return files


def on_srcfile_changed(srcfiles, progress=gr.Progress()):
    from roop.face_util import norm_crop2
    global SELECTION_FACES_DATA, IS_INPUT, input_faces, face_selection, last_image
    
    IS_INPUT = True

    if srcfiles is None or len(srcfiles) < 1:
        return gr.Column.update(visible=False), None, ui.globals.ui_input_thumbs

    thumbs = []
    for f in srcfiles:    
        source_path = f.name
        if source_path.lower().endswith('fsz'):
            progress(0, desc="Retrieving faces from Faceset File", )      
            unzipfolder = '/temp/faceset'
            if os.path.isdir(unzipfolder):
                files = os.listdir(unzipfolder)
                for file in files:
                    os.remove(os.path.join(unzipfolder, file))
            else:
                os.makedirs(unzipfolder)
            util.unzip(source_path, unzipfolder)
            is_first = True
            face_set = FaceSet()
            for file in os.listdir(unzipfolder):
                if file.endswith(".png"):
                    filename = os.path.join(unzipfolder,file)
                    progress.update()
                    SELECTION_FACES_DATA = extract_face_images(filename,  (False, 0))
                    for f in SELECTION_FACES_DATA:
                        face = f[0]
                        face.mask_offsets = (0,0)
                        face_set.faces.append(face)
                        if is_first: 
                            image = util.convert_to_gradio(f[1])
                            ui.globals.ui_input_thumbs.append(image)
                            is_first = False
                        face_set.ref_images.append(get_image_frame(filename))
            if len(face_set.faces) > 0:
                if len(face_set.faces) > 1:
                    face_set.AverageEmbeddings()
                roop.globals.INPUT_FACESETS.append(face_set)
                                        
        elif util.has_image_extension(source_path):
            progress(0, desc="Retrieving faces from image", )      
            roop.globals.source_path = source_path
            SELECTION_FACES_DATA = extract_face_images(roop.globals.source_path,  (False, 0))
            progress(0.5, desc="Retrieving faces from image")
            for f in SELECTION_FACES_DATA:
                face_set = FaceSet()
                face = f[0]
                face.mask_offsets = (0,0)
                face_set.faces.append(face)
                image = util.convert_to_gradio(f[1])
                ui.globals.ui_input_thumbs.append(image)
                roop.globals.INPUT_FACESETS.append(face_set)
                
    progress(1.0)

    # old style with selecting input faces commented out
    # if len(thumbs) < 1:     
    #     return gr.Column.update(visible=False), None, ui.globals.ui_input_thumbs
    # return gr.Column.update(visible=True), thumbs, gr.Gallery.update(visible=True)

    return gr.Column.update(visible=False), None, ui.globals.ui_input_thumbs


def on_select_input_face(evt: gr.SelectData):
    global SELECTED_INPUT_FACE_INDEX

    SELECTED_INPUT_FACE_INDEX = evt.index


def remove_selected_input_face():
    global SELECTED_INPUT_FACE_INDEX

    if len(roop.globals.INPUT_FACESETS) > SELECTED_INPUT_FACE_INDEX:
        f = roop.globals.INPUT_FACESETS.pop(SELECTED_INPUT_FACE_INDEX)
        del f
    if len(ui.globals.ui_input_thumbs) > SELECTED_INPUT_FACE_INDEX:
        f = ui.globals.ui_input_thumbs.pop(SELECTED_INPUT_FACE_INDEX)
        del f

    return ui.globals.ui_input_thumbs

def on_select_target_face(evt: gr.SelectData):
    global SELECTED_TARGET_FACE_INDEX

    SELECTED_TARGET_FACE_INDEX = evt.index

def remove_selected_target_face():
    if len(roop.globals.TARGET_FACES) > SELECTED_TARGET_FACE_INDEX:
        f = roop.globals.TARGET_FACES.pop(SELECTED_TARGET_FACE_INDEX)
        del f
    if len(ui.globals.ui_target_thumbs) > SELECTED_TARGET_FACE_INDEX:
        f = ui.globals.ui_target_thumbs.pop(SELECTED_TARGET_FACE_INDEX)
        del f
    return ui.globals.ui_target_thumbs





def on_use_face_from_selected(files, frame_num):
    global IS_INPUT, SELECTION_FACES_DATA

    IS_INPUT = False
    thumbs = []
    
    roop.globals.target_path = files[selected_preview_index].name
    if util.is_image(roop.globals.target_path) and not roop.globals.target_path.lower().endswith(('gif')):
        SELECTION_FACES_DATA = extract_face_images(roop.globals.target_path,  (False, 0))
        if len(SELECTION_FACES_DATA) > 0:
            for f in SELECTION_FACES_DATA:
                image = util.convert_to_gradio(f[1])
                thumbs.append(image)
        else:
            gr.Info('No faces detected!')
            roop.globals.target_path = None
                
    elif util.is_video(roop.globals.target_path) or roop.globals.target_path.lower().endswith(('gif')):
        selected_frame = frame_num
        SELECTION_FACES_DATA = extract_face_images(roop.globals.target_path, (True, selected_frame))
        if len(SELECTION_FACES_DATA) > 0:
            for f in SELECTION_FACES_DATA:
                image = util.convert_to_gradio(f[1])
                thumbs.append(image)
        else:
            gr.Info('No faces detected!')
            roop.globals.target_path = None

    if len(thumbs) == 1:
        roop.globals.TARGET_FACES.append(SELECTION_FACES_DATA[0][0])
        ui.globals.ui_target_thumbs.append(thumbs[0])
        return gr.Row.update(visible=False), None, ui.globals.ui_target_thumbs, gr.Dropdown.update(value='Selected face')

    return gr.Row.update(visible=True), thumbs, gr.Gallery.update(visible=True), gr.Dropdown.update(visible=True)



def on_select_face(evt: gr.SelectData):  # SelectData is a subclass of EventData
    global SELECTED_FACE_INDEX
    SELECTED_FACE_INDEX = evt.index
    

def on_selected_face():
    global IS_INPUT, SELECTED_FACE_INDEX, SELECTION_FACES_DATA
    
    fd = SELECTION_FACES_DATA[SELECTED_FACE_INDEX]
    image = util.convert_to_gradio(fd[1])
    if IS_INPUT:
        face_set = FaceSet()
        fd[0].mask_offsets = (0,0)
        face_set.faces.append(fd[0])
        roop.globals.INPUT_FACESETS.append(face_set)
        ui.globals.ui_input_thumbs.append(image)
        return ui.globals.ui_input_thumbs, gr.Gallery.update(visible=True), gr.Dropdown.update(visible=True)
    else:
        roop.globals.TARGET_FACES.append(fd[0])
        ui.globals.ui_target_thumbs.append(image)
        return gr.Gallery.update(visible=True), ui.globals.ui_target_thumbs, gr.Dropdown.update(value='Selected face')
    
#        bt_faceselect.click(fn=on_selected_face, outputs=[dynamic_face_selection, face_selection, input_faces, target_faces])

def on_end_face_selection():
    return gr.Column.update(visible=False), None


def on_preview_frame_changed(frame_num, files, fake_preview, enhancer, detection, face_distance, blend_ratio, use_clip, clip_text, no_face_action):
    global SELECTED_INPUT_FACE_INDEX, is_processing

    from roop.core import live_swap

    mask_offsets = (0,0)
    if len(roop.globals.INPUT_FACESETS) > SELECTED_INPUT_FACE_INDEX:
        if not hasattr(roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0], 'mask_offsets'):
            roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0].mask_offsets = mask_offsets
        mask_offsets = roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0].mask_offsets

    if is_processing or files is None or selected_preview_index >= len(files) or frame_num is None:
        return None, mask_offsets[0], mask_offsets[1]

    filename = files[selected_preview_index].name
    # time.sleep(0.3)
    if util.is_video(filename) or filename.lower().endswith('gif'):
        current_frame = get_video_frame(filename, frame_num)
    else:
        current_frame = get_image_frame(filename)
    if current_frame is None:
        return None, mask_offsets[0], mask_offsets[1]
    

    if not fake_preview or len(roop.globals.INPUT_FACESETS) < 1:
        return util.convert_to_gradio(current_frame), mask_offsets[0], mask_offsets[1]

    roop.globals.face_swap_mode = translate_swap_mode(detection)
    roop.globals.selected_enhancer = enhancer
    roop.globals.distance_threshold = face_distance
    roop.globals.blend_ratio = blend_ratio
    roop.globals.no_face_action = index_of_no_face_action(no_face_action)

    if use_clip and clip_text is None or len(clip_text) < 1:
        use_clip = False

    roop.globals.execution_threads = roop.globals.CFG.max_threads
    current_frame = live_swap(current_frame, roop.globals.face_swap_mode, use_clip, clip_text, SELECTED_INPUT_FACE_INDEX)
    if current_frame is None:
        return None, mask_offsets[0], mask_offsets[1] 
    return util.convert_to_gradio(current_frame), mask_offsets[0], mask_offsets[1]


def gen_processing_text(start, end):
    return f'Processing frame range [{start} - {end}]'

def on_set_frame(sender:str, frame_num):
    global selected_preview_index, list_files_process
    
    idx = selected_preview_index
    if list_files_process[idx].endframe == 0:
        return gen_processing_text(0,0)
    
    start = list_files_process[idx].startframe
    end = list_files_process[idx].endframe
    if sender.lower().endswith('start'):
        list_files_process[idx].startframe = min(frame_num, end)
    else:
        list_files_process[idx].endframe = max(frame_num, start)
    
    return gen_processing_text(list_files_process[idx].startframe,list_files_process[idx].endframe)
    


def on_preview_mask(frame_num, files, clip_text):
    from roop.core import preview_mask
    global is_processing

    if is_processing:
        return None
        
    filename = files[selected_preview_index].name
    if util.is_video(filename) or filename.lower().endswith('gif'):
        current_frame = get_video_frame(filename, frame_num)
    else:
        current_frame = get_image_frame(filename)
    if current_frame is None:
        return None

    current_frame = preview_mask(current_frame, clip_text)
    return util.convert_to_gradio(current_frame)


def on_clear_input_faces():
    ui.globals.ui_input_thumbs.clear()
    roop.globals.INPUT_FACESETS.clear()
    return ui.globals.ui_input_thumbs

def on_clear_destfiles():
    roop.globals.TARGET_FACES.clear()
    ui.globals.ui_target_thumbs.clear()
    return ui.globals.ui_target_thumbs    


def index_of_no_face_action(dropdown_text):
    global no_face_choices

    return no_face_choices.index(dropdown_text) 

def translate_swap_mode(dropdown_text):
    if dropdown_text == "Selected face":
        return "selected"
    elif dropdown_text == "First found":
        return "first"
    elif dropdown_text == "All female":
        return "all_female"
    elif dropdown_text == "All male":
        return "all_male"
    
    return "all"



def start_swap( enhancer, detection, keep_frames, skip_audio, face_distance, blend_ratio,
                use_clip, clip_text, processing_method, no_face_action, progress=gr.Progress(track_tqdm=False)):
    from ui.main import prepare_environment
    from roop.core import batch_process
    global is_processing, list_files_process

    if list_files_process is None or len(list_files_process) <= 0:
        return gr.Button.update(variant="primary"), None
    
    if roop.globals.CFG.clear_output:
        shutil.rmtree(roop.globals.output_path)


    prepare_environment()

    roop.globals.selected_enhancer = enhancer
    roop.globals.target_path = None
    roop.globals.distance_threshold = face_distance
    roop.globals.blend_ratio = blend_ratio
    roop.globals.keep_frames = keep_frames
    roop.globals.skip_audio = skip_audio
    roop.globals.face_swap_mode = translate_swap_mode(detection)
    roop.globals.no_face_action = index_of_no_face_action(no_face_action)
    if use_clip and clip_text is None or len(clip_text) < 1:
        use_clip = False
    
    if roop.globals.face_swap_mode == 'selected':
        if len(roop.globals.TARGET_FACES) < 1:
            gr.Error('No Target Face selected!')
            return gr.Button.update(variant="primary"), None

    is_processing = True            
    yield gr.Button.update(variant="secondary"), None
    roop.globals.execution_threads = roop.globals.CFG.max_threads
    roop.globals.video_encoder = roop.globals.CFG.output_video_codec
    roop.globals.video_quality = roop.globals.CFG.video_quality
    roop.globals.max_memory = roop.globals.CFG.memory_limit if roop.globals.CFG.memory_limit > 0 else None

    batch_process(list_files_process, use_clip, clip_text, processing_method == "In-Memory processing", progress)
    is_processing = False
    outdir = pathlib.Path(roop.globals.output_path)
    outfiles = [item for item in outdir.rglob("*") if item.is_file()]
    if len(outfiles) > 0:
        yield gr.Button.update(variant="primary"),gr.Files.update(value=outfiles)
    else:
        yield gr.Button.update(variant="primary"),None


def stop_swap():
    roop.globals.processing = False
    gr.Info('Aborting processing - please wait for the remaining threads to be stopped')


def on_fps_changed(fps):
    global selected_preview_index, list_files_process

    if len(list_files_process) < 1 or list_files_process[selected_preview_index].endframe < 1:
        return
    list_files_process[selected_preview_index].fps = fps


def on_destfiles_changed(destfiles):
    global selected_preview_index, list_files_process

    if destfiles is None or len(destfiles) < 1:
        list_files_process.clear()
        return gr.Slider.update(value=0, maximum=0), ''
    
    for f in destfiles:
        list_files_process.append(ProcessEntry(f.name, 0,0, 0))

    selected_preview_index = 0
    idx = selected_preview_index    
    
    filename = list_files_process[idx].filename
    
    if util.is_video(filename) or filename.lower().endswith('gif'):
        total_frames = get_video_frame_total(filename)
    else:
        total_frames = 0
    list_files_process[idx].endframe = total_frames
    if total_frames > 0:
        return gr.Slider.update(value=0, maximum=total_frames), gen_processing_text(list_files_process[idx].startframe,list_files_process[idx].endframe)
    return gr.Slider.update(value=0, maximum=total_frames), ''
    



def on_destfiles_selected(evt: gr.SelectData):
    global selected_preview_index, list_files_process

    if evt is not None:
        selected_preview_index = evt.index
    idx = selected_preview_index    
    filename = list_files_process[idx].filename
    fps = list_files_process[idx].fps
    if util.is_video(filename) or filename.lower().endswith('gif'):
        total_frames = get_video_frame_total(filename)
        if list_files_process[idx].endframe == 0:
            list_files_process[idx].endframe = total_frames 
    else:
        total_frames = 0
    
    if total_frames > 0:
        return gr.Slider.update(value=list_files_process[idx].startframe, maximum=total_frames), gen_processing_text(list_files_process[idx].startframe,list_files_process[idx].endframe), fps
    return gr.Slider.update(value=0, maximum=total_frames), gen_processing_text(0,0), fps
    
    
    

def on_resultfiles_selected(evt: gr.SelectData, files):
    selected_index = evt.index
    filename = files[selected_index].name
    if util.is_video(filename):
        return gr.update(visible=False), gr.update(visible=True, value=filename)
    else:
        if filename.lower().endswith('gif'):
            current_frame = get_video_frame(filename)
        else:
            current_frame = get_image_frame(filename)
        return gr.update(visible=True, value=util.convert_to_gradio(current_frame)), gr.update(visible=False)



def on_resultfiles_finished(files):
    selected_index = 0
    if files is None or len(files) < 1:
        return None, None
    
    filename = files[selected_index].name
    if util.is_video(filename):
        return gr.update(visible=False), gr.update(visible=True, value=filename)
    else:
        if filename.lower().endswith('gif'):
            current_frame = get_video_frame(filename)
        else:
            current_frame = get_image_frame(filename)
        return gr.update(visible=True, value=util.convert_to_gradio(current_frame)), gr.update(visible=False)
