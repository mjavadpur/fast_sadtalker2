import os
import queue
import threading
import torch 

from gfpgan import GFPGANer

from tqdm import tqdm

from src.utils.videoio import load_video_to_cv2

import cv2
import asyncio


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

async def generate_upscaled_images(images,  restorer):
    
    tasks = [upscale_image(image_path, restorer) for image_path in images]
    return await asyncio.gather(*tasks)
            

async def upscale_image(image, restorer):
    
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cropped_faces, restored_faces, r_img = restorer.enhance(
        img,
        has_aligned=False,
        only_center_face=False,
        paste_back=True)
    r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
    return r_img

'''
2]
2m
# selected audio from exmaple/driven_audio
img = 'examples/source_image/{}.png'.format(default_head_name.value)
print(img)
!python3.8 inference.py --driven_audio ./examples/driven_audio/RD_Radio31_000.wav \
           --source_image {img} \
           --result_dir ./results --still --preprocess full --enhancer gfpgan
account_circle
examples/source_image/full3.png
using safetensor as default
KeypointExtractor 1:  2.2323132870005793
KeypointExtractor 2:  0.6153312200003711
CropAndExtract  Preprocesser:  2.8479465919990616
CropAndExtract  self.net_recon:  0.6934777080005006
CropAndExtract  self.lm3d_std:  0.0008215280013246229
Init Time: 6.380234718322754
3DMM Extraction for source image
landmark Det:: 100% 1/1 [00:00<00:00, 15.58it/s]
3DMM Extraction In Video:: 100% 1/1 [00:00<00:00, 45.96it/s]
mel:: 100% 96/96 [00:00<00:00, 40439.21it/s]
audio2exp:: 100% 10/10 [00:00<00:00, 277.08it/s]
Face Renderer:: 100% 6/6 [00:18<00:00,  3.10s/it]
IMAGEIO FFMPEG_WRITER WARNING: input image is not divisible by macro_block_size=16, resizing from (256, 259) to (256, 272) to ensure video compatibility with most codecs and players. To prevent resizing, make your input image divisible by the macro_block_size or set the macro_block_size to 1 (risking incompatibility).
The generated video is named ./results/2023_12_07_07.27.46/full3##RD_Radio31_000.mp4
OpenCV: FFMPEG: tag 0x5634504d/'MP4V' is not supported with codec id 12 and format 'mp4 / MP4 (MPEG-4 Part 14)'
OpenCV: FFMPEG: fallback to use tag 0x7634706d/'mp4v'
seamlessClone:: 100% 96/96 [00:14<00:00,  6.75it/s]
The generated video is named ./results/2023_12_07_07.27.46/full3##RD_Radio31_000_full.mp4
face enhancer....
The generated video is named ./results/2023_12_07_07.27.46/full3##RD_Radio31_000_enhanced.mp4
Moviepy - Building video ./interpolate_videos.mp4.
MoviePy - Writing audio in interpolate_videosTEMP_MPY_wvf_snd.mp4
MoviePy - Done.
Moviepy - Writing video ./interpolate_videos.mp4

Moviepy - Done !
Moviepy - video ready ./interpolate_videos.mp4
Interpolated result is located in ./interpolate_videos.mp4.
The generated video is named: ./results/2023_12_07_07.27.46.mp4
Extract Time: 1.4975826740264893
Gen Coeff Time: 0.4841935634613037
Render Time: 89.59463238716125
Interpolated Time: 12.924031972885132
 -- Inference exec time: 104.50184750556946
'''