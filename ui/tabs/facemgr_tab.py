import os
import shutil
import cv2
import gradio as gr
import roop.utilities as util
import roop.globals
from roop.face_util import extract_face_images

selected_face_index = -1
thumbs = []
images = []


def facemgr_tab():
    with gr.Tab("👨‍👩‍👧‍👦 Face Management"):
        with gr.Row():
            gr.Markdown("""
                        # Create blending facesets
                        Add multiple reference images into a faceset file.
                        """)
        with gr.Row():
            fb_facesetfile = gr.Files(label='Faceset', file_count='single', file_types=['.fsz'], interactive=True)
            fb_files = gr.Files(label='Input Files', interactive=True)
        with gr.Row():
            with gr.Column():
                gr.Button("👀 Open Output Folder", size='sm').click(fn=lambda: util.open_folder(roop.globals.output_path))
            with gr.Column():
                gr.Markdown(' ')
        with gr.Row():
            faces = gr.Gallery(label="Faces in this Faceset", allow_preview=True, preview=True, height=128, object_fit="scale-down")
        with gr.Row():
            fb_remove = gr.Button("Remove selected", variant='secondary')
            fb_update = gr.Button("Create/Update Faceset file", variant='primary')
            fb_clear = gr.Button("Clear all", variant='stop')

    fb_facesetfile.change(fn=on_faceset_changed, inputs=[fb_facesetfile], outputs=[faces])
    fb_files.change(fn=on_fb_files_changed, inputs=[fb_files], outputs=[faces])
    fb_update.click(fn=on_update_clicked, outputs=[fb_facesetfile])
    fb_remove.click(fn=on_remove_clicked, outputs=[faces])
    fb_clear.click(fn=on_clear_clicked, outputs=[faces, fb_files, fb_facesetfile])
    faces.select(fn=on_face_selected)

def on_faceset_changed(faceset, progress=gr.Progress()):
    global thumbs, images

    if faceset is None:
        return thumbs

    thumbs.clear()
    filename = faceset.name
        
    if filename.lower().endswith('fsz'):
        progress(0, desc="Retrieving faces from Faceset File", )      
        unzipfolder = '/temp/faceset'
        if os.path.isdir(unzipfolder):
            shutil.rmtree(unzipfolder)
        os.makedirs(unzipfolder)
        util.unzip(filename, unzipfolder)
        for file in os.listdir(unzipfolder):
            if file.endswith(".png"):
                SELECTION_FACES_DATA = extract_face_images(os.path.join(unzipfolder,file),  (False, 0), 0.5)
                if len(SELECTION_FACES_DATA) < 1:
                    gr.Warning(f"No face detected in {file}!")
                for f in SELECTION_FACES_DATA:
                    image = f[1]
                    images.append(image)
                    thumbs.append(util.convert_to_gradio(image))
        
        return thumbs


def on_fb_files_changed(inputfiles, progress=gr.Progress()):
    global thumbs, images

    if inputfiles is None or len(inputfiles) < 1:
        return thumbs
    
    progress(0, desc="Retrieving faces from images", )
    for f in inputfiles:
        source_path = f.name
        if util.has_image_extension(source_path):
            roop.globals.source_path = source_path
            SELECTION_FACES_DATA = extract_face_images(roop.globals.source_path,  (False, 0), 0.5)
            for f in SELECTION_FACES_DATA:
                image = f[1]
                images.append(image)
                thumbs.append(util.convert_to_gradio(image))
    return thumbs
    
def on_face_selected(evt: gr.SelectData):
    global selected_face_index

    if evt is not None:
        selected_face_index = evt.index

def on_remove_clicked():
    global thumbs, images, selected_face_index

    if len(thumbs) > selected_face_index:
        f = thumbs.pop(selected_face_index)
        del f
        f = images.pop(selected_face_index)
        del f
    return thumbs

def on_clear_clicked():
    global thumbs, images

    thumbs.clear()
    images.clear()
    return thumbs, None, None





def on_update_clicked():
    if len(images) < 1:
        gr.Warning(f"No faces to create faceset from!")
        return None

    imgnames = []
    for index,img in enumerate(images):
        filename = os.path.join(roop.globals.output_path, f'{index}.png')
        # if img.shape[0] != 512 or img.shape[1] != 512:
        # cv2.imwrite(filename, resize_image_keep_content(img, 512, 512))
        # removed resizing
        cv2.imwrite(filename, img)
        imgnames.append(filename)

    finalzip = os.path.join(roop.globals.output_path, 'faceset.fsz')        
    util.zip(imgnames, finalzip)
    return finalzip


