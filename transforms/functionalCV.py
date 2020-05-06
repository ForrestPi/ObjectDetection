from __future__ import division
import torch
import math
import random
import cv2
import numpy as np
import numbers
import types
import collections
import warnings


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if not _is_numpy_image(pic):
        raise TypeError('pic should be ndarray. Got {}'.format(type(pic)))
    # handle numpy array
    img = torch.from_numpy(pic.transpose((2, 0, 1)))
    # backward compatibility
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img

'''
def to_pil_image(pic, mode=None):
    """Convert a tensor or an ndarray to PIL Image.

    See :class:`~torchvision.transforms.ToPIlImage` for more details.

    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).

    .. _PIL.Image mode: http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#modes

    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if not(_is_numpy_image(pic) or _is_tensor_image(pic)):
        raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

    npimg = pic
    if isinstance(pic, torch.FloatTensor):
        pic = pic.mul(255).byte()
    if torch.is_tensor(pic):
        npimg = np.transpose(pic.numpy(), (1, 2, 0))

    if not isinstance(npimg, np.ndarray):
        raise TypeError('Input pic must be a torch.Tensor or NumPy ndarray, ' +
                        'not {}'.format(type(npimg)))

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = 'L'
        elif npimg.dtype == np.int16:
            expected_mode = 'I;16'
        elif npimg.dtype == np.int32:
            expected_mode = 'I'
        elif npimg.dtype == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError("Incorrect mode ({}) supplied for input type {}. Should be {}"
                             .format(mode, np.dtype, expected_mode))
        mode = expected_mode

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError("Only modes {} are supported for 4D inputs".format(permitted_4_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError("Only modes {} are supported for 3D inputs".format(permitted_3_channel_modes))
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'

    if mode is None:
        raise TypeError('Input type {} is not supported'.format(npimg.dtype))

    return Image.fromarray(npimg, mode=mode)
'''

def normalize(tensor, mean, std):
    """Normalize a tensor image with mean and standard deviation.

    See ``Normalize`` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not _is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')
    # TODO: make efficient
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor

def normalize_caffe(tensor, mean, scale):
    if not _is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')
    for t, m, s in zip(tensor, mean, scale):
        t.sub_(m).mul_(s)
    return tensor



def resize(img, size, interpolation=cv2.INTER_LINEAR):
    """Resize the input CV2 Image to the given size.

    Args:
        img (CV2 Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``CV2.INTER_LINEAR``

    Returns:
        CV2 Image: Resized image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be nparrary Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        h, w = img.shape[:2]
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return cv2.resize(img, (ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return cv2.resize(img, (ow, oh), interpolation)
    else:
        return cv2.resize(img, tuple(size[::-1]), interpolation)


def scale(*args, **kwargs):
    warnings.warn("The use of the transforms.Scale transform is deprecated, " +
                  "please use transforms.Resize instead.")
    return resize(*args, **kwargs)


def pad(img, padding, fill=0, padding_mode='constant'):
    """Pad the given CV2 Image on all sides with speficified padding mode and fill value.

    Args:
        img (CV2 Image): Image to be padded.
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
            constant: pads with a constant value, this value is specified with fill
            edge: pads with the last value on the edge of the image
            reflect: pads with reflection of image (without repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
            symmetric: pads with reflection of image (repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    Returns:
        CV2 Image: Padded image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be nparray Image. Got {}'.format(type(img)))

    if not isinstance(padding, (numbers.Number, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError('Got inappropriate fill arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')

    if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
        raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric'], \
        'Padding mode should be either constant, edge, reflect or symmetric'

    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    if isinstance(padding, collections.Sequence) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    if isinstance(padding, collections.Sequence) and len(padding) == 4:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]

    if padding_mode == 'constant':
        if isinstance(fill, int):
            fill_R = fill_G = fill_B = fill
        else:
            fill_R = fill[0]
            fill_G = fill[1]
            fill_B = fill[2]
        img_b, img_g, img_r = cv2.split(img)
        img_b = np.pad(img_b, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode, constant_values=((fill_B, fill_B),(fill_B, fill_B)))
        img_g = np.pad(img_g, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode, constant_values=((fill_G, fill_G),(fill_G, fill_G)))
        img_r = np.pad(img_r, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode, constant_values=((fill_R, fill_R),(fill_R, fill_R)))
        img = cv2.merge([img_b,img_g,img_r])
    else:
        if len(img.shape) == 3:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), padding_mode)

    return img


def crop(img, i, j, h, w):
    """Crop the given CV2 Image.

    Args:
        img (CV2 Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.

    Returns:
        CV2 Image: Cropped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be nparray Image. Got {}'.format(type(img)))

    return img[i:i+h, j:j+w]


def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    h, w = img.shape[:2]
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, i, j, th, tw)


def resized_crop(img, i, j, h, w, size, interpolation=cv2.INTER_LINEAR):
    """Crop the given CV2 Image and resize it to desired size.

    Notably used in RandomResizedCrop.

    Args:
        img (CV2 Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        size (sequence or int): Desired output size. Same semantics as ``scale``.
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_LINEAR``.
    Returns:
        CV2 Image: Cropped image.
    """
    assert _is_numpy_image(img), 'img should be nparray Image'
    img = crop(img, i, j, h, w)
    img = resize(img, size, interpolation)
    return img


def hflip(img):
    """Horizontally flip the given PIL Image.

    Args:
        img (PIL Image): Image to be flipped.

    Returns:
        PIL Image:  Horizontall flipped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be nparray Image. Got {}'.format(type(img)))

    return cv2.flip(img, 1)


def vflip(img):
    """Vertically flip the given PIL Image.

    Args:
        img (PIL Image): Image to be flipped.

    Returns:
        PIL Image:  Vertically flipped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be nparray Image. Got {}'.format(type(img)))

    return cv2.flip(img, 0)


def five_crop(img, size):
    """Crop the given PIL Image into four corners and the central crop.

    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

    Args:
       size (sequence or int): Desired output size of the crop. If size is an
           int instead of sequence like (h, w), a square crop (size, size) is
           made.
    Returns:
        tuple: tuple (tl, tr, bl, br, center) corresponding top left,
            top right, bottom left, bottom right and center crop.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    # w, h = img.size
    h, w = img.shape[:2]
    crop_h, crop_w = size
    if crop_w > w or crop_h > h:
        raise ValueError("Requested crop size {} is bigger than input size {}".format(size,
                                                                                      (h, w)))
    # img[i:i+h, j:j+w]
    # tl = img.crop((0, 0, crop_w, crop_h))
    tl = img[0:0+crop_h, 0:0+crop_w]
    # tr = img.crop((w - crop_w, 0, w, crop_h))
    tr = img[0:0+crop_h, w-crop_w:]
    # bl = img.crop((0, h - crop_h, crop_w, h))
    bl = img[h-crop_h:, 0:0+crop_w]
    # br = img.crop((w - crop_w, h - crop_h, w, h))
    br = img[h-crop_h:,w-crop_w:]
    center = center_crop(img, (crop_h, crop_w))
    return (tl, tr, bl, br, center)


def ten_crop(img, size, vertical_flip=False):
    """Crop the given PIL Image into four corners and the central crop plus the
       flipped version of these (horizontal flipping is used by default).

    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

       Args:
           size (sequence or int): Desired output size of the crop. If size is an
               int instead of sequence like (h, w), a square crop (size, size) is
               made.
           vertical_flip (bool): Use vertical flipping instead of horizontal

        Returns:
            tuple: tuple (tl, tr, bl, br, center, tl_flip, tr_flip, bl_flip,
                br_flip, center_flip) corresponding top left, top right,
                bottom left, bottom right and center crop and same for the
                flipped image.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    first_five = five_crop(img, size)

    if vertical_flip:
        img = vflip(img)
    else:
        img = hflip(img)

    second_five = five_crop(img, size)
    return first_five + second_five


def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.

    Args:
        img (np.ndarray): CV Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        np.ndarray: Brightness adjusted image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    im = img.astype(np.float32) * brightness_factor
    im = im.clip(min=0, max=255)
    return im.astype(img.dtype)


def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an Image.

    Args:
        img (np.ndarray): CV Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        np.ndarray: Contrast adjusted image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))
    im = img.astype(np.float32)
    mean = round(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).mean())
    im = (1-contrast_factor)*mean + contrast_factor * im
    im = im.clip(min=0, max=255)
    return im.astype(img.dtype)


def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.

    Args:
        img (np.ndarray): CV Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a gray image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        np.ndarray: Saturation adjusted image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    im = img.astype(np.float32)
    degenerate = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    im = (1-saturation_factor) * degenerate + saturation_factor * im
    im = im.clip(min=0, max=255)
    return im.astype(img.dtype)


def adjust_hue(img, hue_factor):
    """Adjust hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See https://en.wikipedia.org/wiki/Hue for more details on Hue.

    Args:
        img (np.ndarray): CV Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        np.ndarray: Hue adjusted image.
    """
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    im = img.astype(np.uint8)
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV_FULL)
    hsv[..., 0] += np.uint8(hue_factor * 255)

    im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)
    return im.astype(img.dtype)


def adjust_gamma(img, gamma, gain=1):
    """Perform gamma correction on an image.

    Also known as Power Law Transform. Intensities in RGB mode are adjusted
    based on the following equation:

        I_out = 255 * gain * ((I_in / 255) ** gamma)

    See https://en.wikipedia.org/wiki/Gamma_correction for more details.

    Args:
        img (CV2 Image): CV2 Image to be adjusted.
        gamma (float): Non negative real number. gamma larger than 1 make the
            shadows darker, while gamma smaller than 1 make dark regions
            lighter.
        gain (float): The constant multiplier.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be nparray Image. Got {}'.format(type(img)))

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    # input_mode = img.mode
    # img = img.convert('RGB')

    # gamma_map = [255 * gain * pow(ele / 255., gamma) for ele in range(256)] * 3
    img = (255 * gain * ((img / 255) ** gamma)).astype(np.uint8)
    # img = img.point(gamma_map)  # use PIL's point-function to accelerate this part

    # img = img.convert(input_mode)
    return img


def rotate(img, angle, resample=False, expand=False, center=None):
    """Rotate the image by angle.


    Args:
        img (CV2 Image): CV2 Image to be rotated.
        angle ({float, int}): In degrees degrees counter clockwise order.
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    if not _is_numpy_image(img):
        raise TypeError('img should be nparray Image. Got {}'.format(type(img)))

    if center == None:
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
    if expand == True:
        ratio = math.sin(2*math.pi*angle/360) * w / h + math.cos(2*math.pi*angle/360)
    else:
        ratio = 1
    M = cv2.getRotationMatrix2D(center, angle, ratio)


    return cv2.warpAffine(img, M, (h, w))


def affine(img, angle=0, translate=(0, 0), scale=1, shear=0, resample='BILINEAR', fillcolor=(0,0,0)):
    """Apply affine transformation on the image keeping image center invariant
    Args:
        img (np.ndarray): PIL Image to be rotated.
        angle ({float, int}): rotation angle in degrees between -180 and 180, clockwise direction.
        translate (list or tuple of integers): horizontal and vertical translations (post-rotation translation)
        scale (float): overall scale
        shear (float): shear angle value in degrees between -180 to 180, clockwise direction.
        resample ({NEAREST, BILINEAR, BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (int or tuple): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
        "Argument translate should be a list or tuple of length 2"

    assert scale > 0.0, "Argument scale should be positive"

    rows, cols, _ = img.shape
    center = (cols * 0.5, rows * 0.5)
    angle = math.radians(angle)
    shear = math.radians(shear)
    M00 = math.cos(angle)*scale
    M01 = -math.sin(angle+shear)*scale
    M10 = math.sin(angle)*scale
    M11 = math.cos(angle+shear)*scale
    M02 = center[0] - center[0]*M00 - center[1]*M01 + translate[0]
    M12 = center[1] - center[0]*M10 - center[1]*M11 + translate[1]
    affine_matrix = np.array([[M00, M01, M02], [M10, M11, M12]], dtype=np.float32)
    dst_img = cv2.warpAffine(img, affine_matrix, (cols, rows), flags=INTER_MODE[resample],
                             borderMode=cv2.BORDER_CONSTANT, borderValue=fillcolor)
    return dst_img


def perspective(img, fov=45, anglex=0, angley=0, anglez=0, shear=0,
                translate=(0, 0), scale=(1, 1), resample='BILINEAR', fillcolor=(0, 0, 0)):
    """

    This function is partly referred to https://blog.csdn.net/dcrmg/article/details/80273818

    """

    imgtype = img.dtype
    h, w, _ = img.shape
    centery = h * 0.5
    centerx = w * 0.5

    alpha = math.radians(shear)
    beta = math.radians(anglez)

    lambda1 = scale[0]
    lambda2 = scale[1]

    tx = translate[0]
    ty = translate[1]

    sina = math.sin(alpha)
    cosa = math.cos(alpha)
    sinb = math.sin(beta)
    cosb = math.cos(beta)

    M00 = cosb * (lambda1 * cosa ** 2 + lambda2 * sina ** 2) - sinb * (lambda2 - lambda1) * sina * cosa
    M01 = - sinb * (lambda1 * sina ** 2 + lambda2 * cosa ** 2) + cosb * (lambda2 - lambda1) * sina * cosa

    M10 = sinb * (lambda1 * cosa ** 2 + lambda2 * sina ** 2) + cosb * (lambda2 - lambda1) * sina * cosa
    M11 = + cosb * (lambda1 * sina ** 2 + lambda2 * cosa ** 2) + sinb * (lambda2 - lambda1) * sina * cosa
    M02 = centerx - M00 * centerx - M01 * centery + tx
    M12 = centery - M10 * centerx - M11 * centery + ty
    affine_matrix = np.array([[M00, M01, M02], [M10, M11, M12], [0, 0, 1]], dtype=np.float32)
    # -------------------------------------------------------------------------------
    z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(math.radians(fov / 2))

    radx = math.radians(anglex)
    rady = math.radians(angley)

    sinx = math.sin(radx)
    cosx = math.cos(radx)
    siny = math.sin(rady)
    cosy = math.cos(rady)

    r = np.array([[cosy, 0, -siny, 0],
                  [-siny * sinx, cosx, -sinx * cosy, 0],
                  [cosx * siny, sinx, cosx * cosy, 0],
                  [0, 0, 0, 1]])

    pcenter = np.array([centerx, centery, 0, 0], np.float32)

    p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
    p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
    p3 = np.array([0, h, 0, 0], np.float32) - pcenter
    p4 = np.array([w, h, 0, 0], np.float32) - pcenter

    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)

    list_dst = [dst1, dst2, dst3, dst4]

    org = np.array([[0, 0],
                    [w, 0],
                    [0, h],
                    [w, h]], np.float32)

    dst = np.zeros((4, 2), np.float32)

    for i in range(4):
        dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
        dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]

    perspective_matrix = cv2.getPerspectiveTransform(org, dst)
    total_matrix = perspective_matrix @ affine_matrix

    result_img = cv2.warpPerspective(img, total_matrix, (w, h), flags=INTER_MODE[resample],
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=fillcolor)
    return result_img.astype(imgtype)


def gaussian_noise(img: np.ndarray, mean, std):
    imgtype = img.dtype
    gauss = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy = np.clip((1 + gauss) * img.astype(np.float32), 0, 255)
    return noisy.astype(imgtype)


def poisson_noise(img):
    imgtype = img.dtype
    img = img.astype(np.float32)/255.0
    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = 255 * np.clip(np.random.poisson(img.astype(np.float32) * vals) / float(vals), 0, 1)
    return noisy.astype(imgtype)


def salt_and_pepper(img, prob=0.01):

    ''' Adds "Salt & Pepper" noise to an image.
        prob: probability (threshold) that controls level of noise
    '''

    imgtype = img.dtype
    rnd = np.random.rand(img.shape[0], img.shape[1])
    noisy = img.copy()
    noisy[rnd < prob/2] = 0.0
    noisy[rnd > 1 - prob/2] = 255.0
    return noisy.astype(imgtype)

'''
def _get_inverse_affine_matrix(center, angle, translate, scale, shear):
    # Helper method to compute inverse matrix for affine transformation

    # As it is explained in PIL.Image.rotate
    # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, scale, shear) = [ cos(a)*scale    -sin(a + shear)*scale     0]
    #                              [ sin(a)*scale    cos(a + shear)*scale     0]
    #                              [     0                  0          1]
    # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

    angle = math.radians(angle)
    shear = math.radians(shear)
    scale = 1.0 / scale

    # Inverted rotation matrix with scale and shear
    d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
    matrix = [
        math.cos(angle + shear), math.sin(angle + shear), 0,
        -math.sin(angle), math.cos(angle), 0
    ]
    matrix = [scale / d * m for m in matrix]

    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
    matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += center[0]
    matrix[5] += center[1]
    return matrix


def affine(img, angle, translate, scale, shear, resample=0, fillcolor=None):
    """Apply affine transformation on the image keeping image center invariant

    Args:
        img (CV2 Image): CV2 Image to be rotated.
        angle ({float, int}): rotation angle in degrees between -180 and 180, clockwise direction.
        translate (list or tuple of integers): horizontal and vertical translations (post-rotation translation)
        scale (float): overall scale
        shear (float): shear angle value in degrees between -180 to 180, clockwise direction.
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
        "Argument translate should be a list or tuple of length 2"

    assert scale > 0.0, "Argument scale should be positive"

    output_size = img.size
    center = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)
    matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
    kwargs = {"fillcolor": fillcolor} if PILLOW_VERSION[0] == '5' else {}
    return img.transform(output_size, Image.AFFINE, matrix, resample, **kwargs)


def to_grayscale(img, num_output_channels=1):
    """Convert image to grayscale version of image.

    Args:
        img (PIL Image): Image to be converted to grayscale.

    Returns:
        PIL Image:  Grayscale version of the image.
                    if num_output_channels == 1 : returned image is single channel
                    if num_output_channels == 3 : returned image is 3 channel with r == g == b
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if num_output_channels == 1:
        img = img.convert('L')
    elif num_output_channels == 3:
        img = img.convert('L')
        np_img = np.array(img, dtype=np.uint8)
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, 'RGB')
    else:
        raise ValueError('num_output_channels should be either 1 or 3')

    return img
'''
