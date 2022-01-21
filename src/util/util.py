import os
import torch

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes=None, strides=None, rates=None, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    if rates is None:
        rates = [1, 1]
    if strides is None:
        strides = [2, 2]
    if ksizes is None:
        ksizes = [3, 3]
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes, dilation=rates, padding=0, stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


def get_similarity_map(low, ref, ksizes, strides=None, rates=None, sigma=10):
    """
    :low: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :ref: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols].
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    if rates is None:
        rates = [1, 1]
    if strides is None:
        strides = [1, 1]
    batch_size, channel, rows, cols = low.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]

    low_patches = extract_image_patches(low, ksizes=ksizes, strides=strides, rates=rates, padding='same')
    ref_patches = extract_image_patches(ref, ksizes=ksizes, strides=strides, rates=rates, padding='same')
    norm = torch.mul(torch.sqrt((low_patches*low_patches).sum(dim=1)),
                     torch.sqrt((ref_patches*ref_patches).sum(dim=1)))
    similarity = torch.div((low_patches*ref_patches).sum(dim=1), norm)
    max_similarity = similarity.max(dim=1)
    similarity_map = torch.softmax(similarity * sigma, dim=1)
    # let max = max_similarity
    for batch in range(batch_size):
        similarity_map[batch] = similarity_map[batch]/similarity_map[batch].max()*max_similarity[0][batch]
    similarity_map_normed = similarity_map.reshape([batch_size, 1, out_rows, out_cols])
    return similarity_map_normed # [N, 1, h, w]