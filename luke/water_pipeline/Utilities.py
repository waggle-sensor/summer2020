import numpy as np
# import cupy as np
from typing import Tuple


def pad_for_blocks(img: np.ndarray, blocksize: int) -> np.ndarray:
    """
    Pad image or frame sequence with repeated columns/rows
    :param img: Shape of ([frame #], y-size, x-size)
    :param blocksize: Integer describing length of block edge
    :return: Same as input shape but with added rows/columns if necessary
    """

    # Add columns
    augmented_ar = img
    if img.shape[-1] % blocksize != 0:
        col_to_add = blocksize - (img.shape[-1] % blocksize)
        augmented_ar = np.concatenate((img, img[..., -col_to_add:]), axis=-1)
    # Add rows
    if img.shape[-2] % blocksize != 0:
        row_to_add = blocksize - (img.shape[-2] % blocksize)
        if len(img.shape) == 2:
            addition = augmented_ar[-row_to_add:, :]
        else:
            addition = augmented_ar[:, -row_to_add:, :]
        augmented_ar = np.concatenate((augmented_ar, addition), axis=-2)
    # Return array with or without changes
    return augmented_ar


def sliding_win_select(ar: np.ndarray, win_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Slides a window across an ndarray, grabbing the contents of the window and placing them into a new ndarray
    win_size shape: (..., y, x)
    returns: (..., flat list of blocks, block size, block size) and int dimensions of (block-y, block-x)
    """

    arr = pad_for_blocks(ar, win_size)

    nrows = ncols = win_size
    h, w = arr.shape[-2:]

    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)

    if len(arr.shape) == 2:
        return (arr.reshape((h // nrows, nrows, -1, ncols))
                .swapaxes(-3, -2)
                .reshape(-1, nrows, ncols)), (h // nrows, w // ncols)
    elif len(arr.shape) == 3:
        frame_n = arr.shape[0]
        return (arr.reshape((frame_n, h // nrows, nrows, w // ncols, ncols))
                .swapaxes(-3, -2)
                .reshape(frame_n, -1, nrows, ncols)), (h // nrows, w // ncols)
    else:
        raise RuntimeError('Unsupported shape')


def get_mask_from_blocks(blocks: np.ndarray, blocksize: int):
    """
    Accepts an ndarray of  (y-blocks, x-blocks) and outputs an array of (y-blocks * blocksize, x-blocks * blocksize)
    """
    mask = np.zeros((blocks.shape[0] * blocksize, blocks.shape[1] * blocksize))
    for y in range(blocks.shape[0]):
        for x in range(blocks.shape[1]):
            mask[y*blocksize:y*blocksize+blocksize, x*blocksize:x*blocksize+blocksize] = blocks[y, x]
    return mask
