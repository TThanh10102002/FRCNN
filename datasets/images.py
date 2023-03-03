import imageio
from PIL import Image
import numpy as np


def compute_scale_factor(org_width, org_height, min_dim_pixels):
    if not min_dim_pixels:
        return 1.0
    if org_width > org_height:
        scale_factor = min_dim_pixels / org_height
    else:
        scale_factor = min_dim_pixels / org_width
    return scale_factor


def preprocess_vgg16(image):
    img_data = image[:,:,::-1]
    img_data[:,:,0] -= 103.939
    img_data[:,:,1] -= 116.779
    img_data[:,:,2] -= 123.680
    return img_data


def load_img(url, min_dim_pixels = None, horizontal_flip = False):
    data = imageio.imread(url, pilmode = 'RGB')
    img = Image.fromarray(data, mode='RGB')
    org_width, org_height = img.width, img.height
    if horizontal_flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    if min_dim_pixels is not None:
        scale_factor = compute_scale_factor(org_width=org_width, org_height=org_height, min_dim_pixels=min_dim_pixels)
        width = int(img.width * scale_factor)
        height = int(img.height * scale_factor)
        img = img.resize((width, height), resample=Image.BILINEAR)
    else:
        scale_factor = 1.0

    img_data = np.array(img).astype(np.float32)
    img_data = preprocess_vgg16(image=img_data)
    return img_data, img, scale_factor, (img.shape[0], org_height, org_width)
