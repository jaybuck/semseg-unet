"""Functions needed by unet2.py, unet_predict.py and others"""
import os
import sys
from pathlib import Path
from functools import reduce

import numpy as np
import OpenImageIO as oiio

# if sys.platform == 'darwin':
#     import OpenImageIO as oiio
# else:
#     import PyOpenImageIO as oiio


def get_numpy_var_info(npvar: np.ndarray):
    info_str = 'dtype: {}  shape: {}    mean: {} min: {} max: {}'.format(npvar.dtype, npvar.shape,
                                                                        npvar.mean(), npvar.min(), npvar.max())
    return info_str


def get_imagebuf_info(buf):
    spec = buf.spec()
    spec_str = 'w: {}  h: {}  nchannels: {}  format: {}   channelnames: {}'.format(spec.width,
                                                                              spec.height,
                                                                              spec.nchannels,
                                                                              spec.format,
                                                                              spec.channelnames)
    return spec_str


def listdir_files(dirname, extensions=['.exr']):
    filename_dict = {}
    basepath = Path(dirname)
    dir_entries = basepath.iterdir()
    for item in dir_entries:
        if item.is_file() and item.suffix.lower() in extensions:
            filename_dict[item.name] = str(item)
    return filename_dict


def compute_interval_bounds(begin, desired_length, max_length):
    """
    Computes the beginning and end of an interval bound given that the interval is at most max_length.  It is assumed
    that begin is >= 0.  If begin and begin + desired_length is between [0, max_length), then this pair is returned.
    If begin + desired_length is greater than max_length, then begin is shifted to ensure that desired_length can
    fit.
    :param begin:
    :param desired_length:
    :param max_length:
    :return: (begin, end) interval bounds.  The interval is begin inclusive to end exclusive.  i.e., [begin, end)
    """
    end = begin + desired_length
    if end <= max_length:
        return begin, end
    return max(0, max_length - desired_length), max_length


def convert_to_rgb_image(exr_img):
    """
    Converts the exr_img as a color RGB image or if the exr_img only has a Y channel, then extracts that Y channel
    as an RGB image
    :param exr_img: oiio.ImageBuf object
    :return: oiio.ImageBuf object with 3 channels 'R', 'G', 'B'
    """
    if all([c in exr_img.spec().channelnames for c in ['R', 'G', 'B']]):
        # exr_img is a color image
        return select_channels(exr_img, 'R', 'G', 'B')
    if 'Y' in exr_img.spec().channelnames:
        # luminance image
        luminance = select_channels(exr_img, 'Y')
        img = select_channels_by_index(luminance, [0, 0, 0], ['R', 'G', 'B'])
        return img
    return None

def select_channels_by_index(img, channel_indexes, channel_names):
    """
    Selects the channels as specified by channel_indexes in img and outputs an oiio.ImageBuf object with those
    channels named by channel_names
    :param img: oiio.ImageBuf object
    :param channel_indexes: list of channel indexes or a single int
    :param channel_names: list of channel names (must be the same length as channel_indexes or a single string
    :return: oiio.ImageBuf object
    """
    if isinstance(channel_indexes, int) and isinstance(channel_names, str):
        channel_indexes = (channel_indexes,)
        channel_names = (channel_names,)
    else:
        assert len(channel_indexes) == len(channel_names), \
            "channel_indexes (len={}) must have same length as channel_names (len={})".format(
                len(channel_indexes), len(channel_names))
        channel_indexes = tuple(channel_indexes)
        channel_names = tuple(channel_names)
    dst = oiio.ImageBuf()
    oiio.ImageBufAlgo.channels(dst, img, channelorder=channel_indexes, newchannelnames=channel_names)
    return dst


def select_channels(img, *channel_names):
    """
    Selects the channel names as listed in channel_names and returns an oiio.ImageBuf object with only those channels
    in it.
    :param img: oiio.ImageBuf object
    :param channel_names: list of channel names
    :return: oiio.ImageBuf object containing only the channels specified in channel_names
    """
    dst = oiio.ImageBuf()
    oiio.ImageBufAlgo.channels(dst, img, tuple(channel_names))
    return dst


def concatenate_images(*imgs):
    """
    Concatenate the channels of each image (oiio.ImageBuf object) passed in and output a single oiio.ImageBuf object
    with all channels concatenates into one.
    :param imgs: list of oiio.ImageBuf object
    :return: oiio.ImageBuf object
    """
    if len(imgs) == 0:
        return None
    elif len(imgs) == 1:
        return imgs[0]

    def _append_imgs(a, b):
        dst = oiio.ImageBuf()
        oiio.ImageBufAlgo.channel_append(dst, a, b)
        return dst

    return reduce(lambda a, b: _append_imgs(a, b), imgs)


def nparray_to_image_buf(array, img_buf_type=oiio.UINT8, channel_names=None):
    """
    Convert the numpy array to an oiio.ImageBuf object
    :param array: numpy array (should a 3-d array where the last dimension is the pixel channels)
    :param img_buf_type: (default: oiio.UINT8)
    :param channel_names: (default: None) to specify explicit channel names
    :return: oiio.ImageBuf object
    """
    if len(array.shape) == 2:
        array = np.reshape(array, (array.shape[0], array.shape[1], 1))
    width = array.shape[1]
    height = array.shape[0]
    # depth = array.shape[2] if len(array.shape) == 3 else 1
    depth = array.shape[2]
    # print('nparray_to_image_buf: width {}   height {}   depth {}  img_buf_type {}'.format(width, height, depth, img_buf_type))
    dst = oiio.ImageBuf(oiio.ImageSpec(width, height, depth, img_buf_type))
    if channel_names:
        assert isinstance(channel_names, tuple), 'channel_names must be tuple'
        assert len(channel_names) == dst.spec().nchannels, \
            'channel_names must be a tuple of length {}'.format(dst.spec().nchannels)
        dst.spec().channelnames = channel_names
    if not dst.set_pixels(oiio.ROI(), array):
        raise RuntimeError('Error creating ImageBuf: {}'.format(dst.geterror()))
    return dst


def color_img(rgb_img, mask, thresh, color):
    """
    Color the image by color for pixels where the mask is greater than thresh
    :param rgb_img: 3-d ndarray (w x h x numchannels)
    :param mask: 2-d ndarray (w x h)
    :param thresh: threshold
    :param color: array-like of length 3 indicating (r, g, b) pixel values
    :return: colored image
    """
    max_pix = rgb_img.max(axis=1)
    max_pixel = max_pix.max(axis=0)
    # print('color_img: rgb_img in:  dtype: {}   shape: {}  max: {}'.format(rgb_img.dtype, rgb_img.shape, max_pixel))

    mask = mask > thresh
    mask = mask.astype(np.uint8)
    # print('color_img:  mask:  mean: {} max: {}  dtype: {}   shape: {}'.format(mask.mean(), mask.max(), mask.dtype, mask.shape))

    color2 = np.array(color, ndmin=2)
    # print('color2 dtype: {}   shape: {}'.format(color2.dtype, color2.shape))

    mask2 = np.dot(mask, np.array(color, ndmin=2))
    # print('mask2 dtype: {}   shape: {}'.format(mask2.dtype, mask2.shape))

    rgb_img = np.maximum(rgb_img, mask2)
    rgb_img = rgb_img.astype(np.uint8)
    # print('rgb_img dtype: {}   shape: {}'.format(rgb_img.dtype, rgb_img.shape))

    return rgb_img


def convert_to_rgb_image(exr_img):
    """
    Converts the exr_img as a color RGB image or if the exr_img only has a Y channel, then extracts that Y channel
    as an RGB image
    :param exr_img: oiio.ImageBuf object
    :return: oiio.ImageBuf object with 3 channels 'R', 'G', 'B'
    """
    if all([c in exr_img.spec().channelnames for c in ['R', 'G', 'B']]):
        # exr_img is a color image
        return select_channels(exr_img, 'R', 'G', 'B')
    if 'Y' in exr_img.spec().channelnames:
        # luminance image
        luminance = select_channels(exr_img, 'Y')
        img = select_channels_by_index(luminance, [0, 0, 0], ['R', 'G', 'B'])
        return img
    return None




