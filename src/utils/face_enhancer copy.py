import os
import torch 

from gfpgan import GFPGANer

from tqdm import tqdm

from src.utils.videoio import load_video_to_cv2

import cv2

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
    # restorer = GFPGANer(
    #     model_path=model_path,
    #     upscale=upscale,
    #     arch=arch,
    #     channel_multiplier=channel_multiplier,
    #     bg_upsampler=bg_upsampler)
    
    iterations = len(images)
    
    # تعداد پردازش‌ها
    processes = 2
    
    # تقسیم تکرارها به چندین گروه
    chunk_size = iterations // processes
    
    ctx = torch.multiprocessing.get_context("spawn")
    # ایجاد یک پول چندپردازشی
    with ctx.Pool(processes=processes) as pool:
        # ایجاد یک ژنراتور ورودی
        input_generator = generator_function(iterations)
        
        # تقسیم ژنراتور به چندین گروه
        chunks = [list(input_generator)[i:i + chunk_size] for i in range(0, iterations, chunk_size)]
        
        # استفاده از imap به جای map برای تابع با yield
        results = pool.imap(process_generator_chunk, [(chunk,model_path, upscale, arch, channel_multiplier, bg_upsampler) for chunk in chunks])
        
        for r_img in results:
            yield r_img
        
    # with ctx.Pool(2) as pool:
    #     r_img = pool.map(restore_mul, images)
    # pool = ctx.Pool(7)
    # pool.map(restore_mul, images)
    
    # ------------------------ restore ------------------------
    # for idx in tqdm(range(len(images)), 'Face Enhancer:'):
        
    #     img = cv2.cvtColor(images[idx], cv2.COLOR_RGB2BGR)
    #      # mj inserted 3 line code
    #     # h, w = img.shape[0:2]
    #     # if h < 300:
    #     #     img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    #     # restore faces and background if necessary
    #     cropped_faces, restored_faces, r_img = restorer.enhance(
    #         img,
    #         has_aligned=False,
    #         only_center_face=False,
    #         paste_back=True)
    #     # mj inserted 3 line code
    #     # interpolation = cv2.INTER_AREA if upscale < 2 else cv2.INTER_LANCZOS4
    #     # h, w = img.shape[0:2]
    #     # r_img = cv2.resize(r_img, (int(w * upscale / 2), int(h * upscale / 2)), interpolation=interpolation)
        
    #     r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
    #     yield r_img
def generator_function(imgs):
    for i in imgs:
        yield i   

def process_generator_chunk(args):
    chunk, model_path, upscale, arch, channel_multiplier, bg_upsampler = args
    results = []
    restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)
    for image in chunk:
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # mj inserted 3 line code
        # h, w = img.shape[0:2]
        # if h < 300:
        #     img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        # restore faces and background if necessary
        cropped_faces, restored_faces, r_img = restorer.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True)
        # mj inserted 3 line code
        # interpolation = cv2.INTER_AREA if upscale < 2 else cv2.INTER_LANCZOS4
        # h, w = img.shape[0:2]
        # r_img = cv2.resize(r_img, (int(w * upscale / 2), int(h * upscale / 2)), interpolation=interpolation)
    
        r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
        results.append(r_img)
    yield results

   
def restore_mul(restorer, image):
    
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # mj inserted 3 line code
    # h, w = img.shape[0:2]
    # if h < 300:
    #     img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    # restore faces and background if necessary
    cropped_faces, restored_faces, r_img = restorer.enhance(
        image,
        has_aligned=False,
        only_center_face=False,
        paste_back=True)
    # mj inserted 3 line code
    # interpolation = cv2.INTER_AREA if upscale < 2 else cv2.INTER_LANCZOS4
    # h, w = img.shape[0:2]
    # r_img = cv2.resize(r_img, (int(w * upscale / 2), int(h * upscale / 2)), interpolation=interpolation)
    
    r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
    # # read image
    # img_name = os.path.basename(img_path)
    # print(f'Processing {img_name} ...')
    # basename, ext = os.path.splitext(img_name)
    # input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # # restore faces and background if necessary
    # cropped_faces, restored_faces, restored_img = restorer.enhance(
    #     input_img,
    #     has_aligned=False,
    #     only_center_face=False,
    #     paste_back=True)
    # # save restored img
    # if restored_img is not None:
    #     if args.ext == 'auto':
    #         extension = ext[1:]
    #     else:
    #         extension = args.ext

    #     if args.suffix is not None:
    #         save_restore_path = os.path.join(args.output, 'restored_imgs', f'{basename}_{args.suffix}.{extension}')
    #     else:
    #         save_restore_path = os.path.join(args.output, 'restored_imgs', f'{basename}.{extension}')
    #     imwrite(restored_img, save_restore_path)