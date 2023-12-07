import os
import queue
import threading
import torch 

from gfpgan import GFPGANer

from tqdm import tqdm

from src.utils.videoio import load_video_to_cv2

import cv2
import asyncio
import concurrent


class GeneratorWithLen(object):
    """ From https://stackoverflow.com/a/7460929 """

    def __init__(self, gen, length):
        self.gen = gen
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen

def enhancer_list(images, method='gfpgan', bg_upsampler='realesrgan'):
    gen = enhancer_generator_no_len(images, method=method, bg_upsampler=bg_upsampler)
    return list(gen)

def enhancer_generator_with_len(images, method='gfpgan', bg_upsampler='realesrgan'):
    """ Provide a generator with a __len__ method so that it can passed to functions that
    call len()"""

    if os.path.isfile(images): # handle video to images
        # TODO: Create a generator version of load_video_to_cv2
        images = load_video_to_cv2(images)

    gen = enhancer_generator_no_len(images, method=method, bg_upsampler=bg_upsampler)
    gen_with_len = GeneratorWithLen(gen, len(images))
    return gen_with_len

def enhancer_generator_no_len(images, method='gfpgan', bg_upsampler='realesrgan'):
    """ Provide a generator function so that all of the enhanced images don't need
    to be stored in memory at the same time. This can save tons of RAM compared to
    the enhancer function. """

    print('face enhancer....')
    if not isinstance(images, list) and os.path.isfile(images): # handle video to images
        images = load_video_to_cv2(images)

    # ------------------------ set up GFPGAN restorer ------------------------
    if  method == 'gfpgan':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.4'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    elif method == 'RestoreFormer':
        arch = 'RestoreFormer'
        channel_multiplier = 2
        model_name = 'RestoreFormer'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
    elif method == 'codeformer': # TODO:
        arch = 'CodeFormer'
        channel_multiplier = 2
        model_name = 'CodeFormer'
        url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
    else:
        raise ValueError(f'Wrong model version {method}.')


    # ------------------------ set up background upsampler ------------------------
    if bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                          'If you really want to use it, please modify the corresponding codes.')
            bg_upsampler = None
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True)  # need to set False in CPU mode
    else:
        bg_upsampler = None

    # determine model paths
    model_path = os.path.join('gfpgan/weights', model_name + '.pth')
    
    if not os.path.isfile(model_path):
        model_path = os.path.join('checkpoints', model_name + '.pth')
    
    if not os.path.isfile(model_path):
        # download pre-trained models from url
        model_path = url

    upscale=2 # or 4 for high resolution videos
    restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(generate_upscaled_images(images,  restorer))

   
    for r_img in results:
        yield r_img
    return r_img

def generate_upscaled_images(images,  restorer):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # برای هر تصویر یک ترد آغاز کنید
        future_to_image = {executor.submit(upscale_image, image, restorer): image for image in images}
        
        for future in concurrent.futures.as_completed(future_to_image):
            img = future_to_image[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing image {img}: {e.with_traceback()}")
    
    return results
    # tasks = [upscale_image(image_path, restorer) for image_path in images]
    # return await asyncio.gather(*tasks)
            

def upscale_image(image, restorer):
    
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    print("img.shape :" + str(img.shape))
    cropped_faces, restored_faces, r_img = restorer.enhance(
        img,
        has_aligned=False,
        only_center_face=False,
        paste_back=True)
    print("r_img.shape:" +str(r_img.shape))
    r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
    return r_img

'''
examples/source_image/full3.png
using safetensor as default
KeypointExtractor 1:  2.204563412000425
KeypointExtractor 2:  0.9296646779985167
CropAndExtract  Preprocesser:  3.134502340000836
CropAndExtract  self.net_recon:  0.6866718259989284
CropAndExtract  self.lm3d_std:  0.0007590149998577544
Init Time: 6.255885362625122
3DMM Extraction for source image
landmark Det:: 100% 1/1 [00:00<00:00, 15.79it/s]
3DMM Extraction In Video:: 100% 1/1 [00:00<00:00, 55.17it/s]
mel:: 100% 96/96 [00:00<00:00, 40581.86it/s]
audio2exp:: 100% 10/10 [00:00<00:00, 279.47it/s]
Face Renderer:: 100% 6/6 [00:18<00:00,  3.08s/it]
IMAGEIO FFMPEG_WRITER WARNING: input image is not divisible by macro_block_size=16, resizing from (256, 259) to (256, 272) to ensure video compatibility with most codecs and players. To prevent resizing, make your input image divisible by the macro_block_size or set the macro_block_size to 1 (risking incompatibility).
The generated video is named ./results/2023_12_07_08.01.58/full3##RD_Radio31_000.mp4
OpenCV: FFMPEG: tag 0x5634504d/'MP4V' is not supported with codec id 12 and format 'mp4 / MP4 (MPEG-4 Part 14)'
OpenCV: FFMPEG: fallback to use tag 0x7634706d/'mp4v'
seamlessClone:: 100% 96/96 [00:14<00:00,  6.77it/s]
The generated video is named ./results/2023_12_07_08.01.58/full3##RD_Radio31_000_full.mp4
face enhancer....
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
r_img.shape:(1536, 1024, 3)
img.shape :(768, 512, 3)
r_img.shape:(1536, 1024, 3)
img.shape :(768, 512, 3)
r_img.shape:(1536, 1024, 3)
img.shape :(768, 512, 3)
r_img.shape:(1536, 1024, 3)
img.shape :(768, 512, 3)
r_img.shape:(1536, 1024, 3)
img.shape :(768, 512, 3)
r_img.shape:(1536, 1024, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
r_img.shape:(1536, 1024, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
r_img.shape:(1536, 1024, 3)
img.shape :(768, 512, 3)
r_img.shape:(1536, 1024, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
r_img.shape:(1536, 1024, 3)
img.shape :(768, 512, 3)
r_img.shape:(1536, 1024, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
r_img.shape:(1536, 1024, 3)
img.shape :(768, 512, 3)
r_img.shape:(1536, 1024, 3)
img.shape :(768, 512, 3)
r_img.shape:(1536, 1024, 3)
img.shape :(768, 512, 3)
r_img.shape:(1536, 1024, 3)
img.shape :(768, 512, 3)
r_img.shape:(1536, 1024, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
r_img.shape:(1536, 1024, 3)
img.shape :(768, 512, 3)
r_img.shape:(1536, 1024, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
r_img.shape:(1536, 1024, 3)
img.shape :(768, 512, 3)
r_img.shape:(1536, 1024, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
r_img.shape:(1536, 1024, 3)
img.shape :(768, 512, 3)
r_img.shape:(1536, 1024, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
r_img.shape:(1536, 1024, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
img.shape :(768, 512, 3)
error in enhancer_generator_with_len
Traceback (most recent call last):
  File "/content/fast_sadtalker2/src/utils/face_enhancer.py", line 131, in generate_upscaled_images
    result = future.result()
  File "/usr/lib/python3.8/concurrent/futures/_base.py", line 437, in result
    return self.__get_result()
  File "/usr/lib/python3.8/concurrent/futures/_base.py", line 389, in __get_result
    raise self._exception
  File "/usr/lib/python3.8/concurrent/futures/thread.py", line 57, in run
    result = self.fn(*self.args, **self.kwargs)
  File "/content/fast_sadtalker2/src/utils/face_enhancer.py", line 145, in upscale_image
    cropped_faces, restored_faces, r_img = restorer.enhance(
  File "/usr/local/lib/python3.8/dist-packages/torch/autograd/grad_mode.py", line 27, in decorate_context
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.8/dist-packages/gfpgan/utils.py", line 145, in enhance
    restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img)
  File "/usr/local/lib/python3.8/dist-packages/facexlib/utils/face_restoration_helper.py", line 291, in paste_faces_to_input_image
    assert len(self.restored_faces) == len(
AssertionError: length of restored_faces and affine_matrices are different.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/content/fast_sadtalker2/src/facerender/animate.py", line 262, in generate
    imageio.mimsave(enhanced_path, enhanced_images_gen_with_len, fps=float(fps))
  File "/usr/local/lib/python3.8/dist-packages/imageio/v2.py", line 331, in mimwrite
    return file.write(ims, **kwargs)
  File "/usr/local/lib/python3.8/dist-packages/imageio/core/legacy_plugin_wrapper.py", line 188, in write
    for written, ndimage in enumerate(ndimage):
  File "/content/fast_sadtalker2/src/utils/face_enhancer.py", line 115, in enhancer_generator_no_len
    results = loop.run_until_complete(generate_upscaled_images(images,  restorer))
  File "/content/fast_sadtalker2/src/utils/face_enhancer.py", line 134, in generate_upscaled_images
    print(f"Error processing image {img}: {e.with_traceback()}")
TypeError: with_traceback() takes exactly one argument (0 given)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "inference.py", line 222, in <module>
    main(args)
  File "inference.py", line 110, in main
    result = args.animate_from_coeff.generate(data, save_dir, pic_path, crop_info, args.fps, \
  File "/content/fast_sadtalker2/src/facerender/animate.py", line 265, in generate
    print(e.message)
AttributeError: 'TypeError' object has no attribute 'message'
'''

