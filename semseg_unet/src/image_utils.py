# Utility functions to handle images

import os
import numpy as np
from functools import reduce
import scipy.misc
import yaml
import glob

import OpenImageIO as oiio

from vision.ml.utils.utils import Utils, list_arg

SUPPORTED_IMG_EXTENSIONS = ['jpg', 'JPG', 'jpeg', 'JPEG',
                            'png', 'PNG',
                            'exr', 'EXR',
                            'tiff', 'TIFF', 'psd']


def find_image_paths(data_paths, root_paths=[]):
    """
    Returns all paths to files with extension in SUPPORTED_IMG_EXTENSIONS under the root path
    :param data_paths: list of image files, or dataset .yaml files, or directory paths (which will be searched
                      recursively for image files or dataset .yaml files)
    :param root_paths: list of directory paths used to resolve image files that are listed in any dataset .yaml files.
    :return: list of file paths to images
    """
    if isinstance(data_paths, str):
        data_paths = [data_paths]
    dataset_yaml_paths = set([])
    img_paths = set([])
    for path in data_paths:
        if not os.path.exists(path):
            print('Path does not exist: {}'.format(path))
            continue
        if os.path.isfile(path):
            if any([path.endswith(ext) for ext in SUPPORTED_IMG_EXTENSIONS]):
                img_paths.add(path)
            elif path.endswith('yaml'):
                dataset_yaml_paths.add(path)
            else:
                print('Ignoring path: {}'.format(path))
        else:
            img_paths.update(find_image_paths(Utils.listdir_recursive(path), root_paths))
    # Resolve paths for any dataset yaml files found
    for dataset_yaml_path in dataset_yaml_paths:
        with open(dataset_yaml_path, 'r') as f:
            dataset = yaml.safe_load(f)
        dataset_img_prefixes = list(dataset.keys())
        for img_prefix in dataset_img_prefixes:
            found = False
            for root_path in root_paths:
                path_prefix = os.path.join(root_path, img_prefix + '*')
                for path in glob.glob(path_prefix):
                    img_paths.add(path)
                    found = True
            if not found:
                print('Could not find image {} specified in {}'.format(img_prefix, dataset_yaml_path))
    return list(img_paths)


def save_output_exr(out_path, img_buf, output_prob, channel_name='class_prob'):
    """
    Save the img_buf and output_prob mask as an exr file
    :param out_path: file path to write to
    :param img_buf: oiio.ImageBuf of original image
    :param output_prob: 2-d ndarray of the output probabilities
    :param channel_name: (default: 'class_prob') name of the channel that output_prob goes into
    """
    output_prob_buf = nparray_to_image_buf(output_prob.reshape(*output_prob.shape, 1), img_buf_type=oiio.UINT8)
    class_prob = select_channels_by_index(output_prob_buf, 0, channel_name)
    out_exr = concatenate_images(img_buf, class_prob)
    out_exr.set_write_format(oiio.HALF)
    out_exr.write(out_path)


def save_output_jpg(out_path, img_buf, output_prob, thresh=.5):
    """
    Save the img_buf and output_prob mask as a jpg file
    :param out_path: file path to write to
    :param image_buf: oiio.ImageBuf of original image
    :param output_prob: 2-d ndarray of the output probabilities
    :param thresh: (default: .5) threshold for the output_prob
    :return:
    """
    img = img_buf.get_pixels(oiio.UINT8)
    img = color_img(img, output_prob.reshape(*output_prob.shape, 1), thresh, [255, 0, 0])
    scipy.misc.imsave(out_path, img)


def color_img(rgb_img, mask, thresh, color):
    """
    Color the image by color for pixels where the mask is greater than thresh
    :param rgb_img: 3-d ndarray (w x h x numchannels)
    :param mask: 2-d ndarray (w x h)
    :param thresh: threshold
    :param color: array-like of length 3 indicating (r, g, b) pixel values
    :return: colored image
    """
    mask = mask > thresh
    mask = np.dot(mask, np.array(color, ndmin=2))
    rgb_img = np.maximum(rgb_img, mask)
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


@list_arg('img')
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


@list_arg('img')
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


@list_arg('array')
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
    depth = array.shape[2] if len(array.shape) == 3 else 1
    dst = oiio.ImageBuf(oiio.ImageSpec(width, height, depth, img_buf_type))
    if channel_names:
        assert isinstance(channel_names, tuple), 'channel_names must be tuple'
        assert len(channel_names) == dst.spec().nchannels, \
            'channel_names must be a tuple of length {}'.format(dst.spec().nchannels)
        dst.spec().channelnames = channel_names
    if not dst.set_pixels(oiio.ROI(), array):
        raise RuntimeError('Error creating ImageBuf: {}'.format(dst.geterror()))
    return dst


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


def crop_bounds_generator(img_shape, crop_shape, stride_step, row_offset=0, col_offset=0):
    """
    Returns a generator that returns the bounds (row_begin, row_end, col_begin, col_end).
    :param img_shape: 2-tuple of the shape of the image (num_rows, num_cols)
    :param crop_shape: 2-tuple of the shape of the crops (num_rows, num_cols)
    :param stride_step: 2-tuple of the steps to take for generating crops (vertical_stride, horizontal_stride)
    :param row_offset: (default: 0) starting offset for row
    :param col_offset: (default: 0) starting offset for column
    :return: returns a generator that yields 4-tuple of (row_begin, row_end, col_begin, col_end)
    """
    def generator():
        row_begin = row_offset
        row_end = 0
        while row_begin < img_shape[0] and row_end < img_shape[0]:
            row_begin, row_end = compute_interval_bounds(row_begin, crop_shape[0], img_shape[0])
            col_begin = col_offset
            col_end = 0
            while col_begin < img_shape[1] and col_end < img_shape[1]:
                col_begin, col_end = compute_interval_bounds(col_begin, crop_shape[1], img_shape[1])
                yield row_begin, row_end, col_begin, col_end
                col_begin += stride_step[1]
            row_begin += stride_step[0]
    return generator
