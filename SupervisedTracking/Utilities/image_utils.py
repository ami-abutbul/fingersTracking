import random

import numpy as np
from PIL import Image


# def url_to_gif(url, dir):
#     return wget.download(url, dir)


def gif_to_images(gif_path, out_dir_res=None):
    im = Image.open(gif_path)
    palette = im.getpalette()
    i = 0
    frames = []
    try:
        while 1:
            im.putpalette(palette)
            new_im = Image.new("RGBA", im.size)
            new_im.paste(im)
            if out_dir_res is not None:
                new_im.save(out_dir_res + '/frame_'+str(i)+'.png')
            frames.append(new_im)
            i += 1
            im.seek(im.tell() + 1)
    except EOFError:
        pass # end of sequence
    return frames


def image_size(path):
    with Image.open(path) as img:
        width, height = img.size
        return width, height


def image_resize(img, out_path=None, size=(512, 512)):
    img = img.resize(size)
    if out_path is not None:
        img.save(out_path, "JPEG")
    return img


def image_thumbnail(img, out_path=None, size=(256, 256)):
    img = img.thumbnail(size, Image.ANTIALIAS)
    if out_path is not None:
        img.save(out_path, "JPEG")
    return img


def concatenate_images_horizontally(imgs_list):
    widths, heights = zip(*(i.size for i in imgs_list))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in imgs_list:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    return new_im


def crop_image_in_middle(image_path, axis='width'):
    img = Image.open(image_path)
    width, height = img.size
    if axis == 'width':
        area1 = (0, 0, width / 2, height)
        area2 = (width / 2, 0, width, height)
        return img.crop(area1), img.crop(area2)
    elif axis == 'height':
        area1 = (0, 0, width, height / 2)
        area2 = (0, height / 2, width, height)
        return img.crop(area1), img.crop(area2)
    else:
        raise Exception("crop_image_in_middle: axis can get one of the values: width, height")


def image_to_mat(image, shape=(256, 256, 3)):
    return np.reshape(np.array(image.getdata()), shape)


def mat_to_image(mat_image, shape=(256, 256, 3)):
    if len(mat_image.shape) != 3:
        mat_image = np.reshape(mat_image, shape)
    return Image.fromarray(mat_image.astype('uint8'), 'RGB')


def channel_to_image(channel, shape=(256, 256, 1)):
    if len(channel.shape) > 3:
        channel = np.reshape(channel, shape)
    channel_r = channel
    channel_g = np.copy(channel)
    channel_b = np.copy(channel)
    image = np.concatenate((channel_r, channel_g, channel_b), axis=2)
    return mat_to_image(image, (shape[0], shape[1], -1))


def make_square_image(im, new_size=512, fill_color=(0, 0, 0, 0)):
    width, height = im.size
    size = max(width, height)
    new_width = int((float(width) / size) * new_size)
    new_height = int((float(height) / size) * new_size)
    im = image_resize(im, size=(new_width, new_height))
    new_im = Image.new('RGBA', (new_size, new_size), fill_color)
    new_im.paste(im, (int((new_size - new_width) / 2), int((new_size - new_height) / 2)))
    return new_im


def crop_random_tile(images, size=(256, 256)):
    dx = size[0]
    dy = size[1]
    res = []

    w, h = images[0].size
    x = random.randint(0, w - dx - 1)
    y = random.randint(0, h - dy - 1)

    for image in images:
        res.append(image.crop((x, y, x + dx, y + dy)))
    return res


# use this function with numpy vectorize
def partial_seg(full_seg_pix, mapping):
    return mapping[full_seg_pix]


def mask_to_rgb(mask, colors):
    return np.array(list(map(lambda x: colors[x], mask)))

if __name__ == '__main__':
    # im1, im2 = crop_image_in_middle('D:/private/facades.tar/facades/train/1.jpg')
    # im1.show()
    # red, _, _ = im1.split()
    # print(type(image_to_mat(red, (256, 256, 1))))
    # # im2.show()
    # # concatenate_images_horizontally([im1, im2]).show()
    # mat1 = image_to_mat(im1)
    # mat2 = image_to_mat(im1)
    #
    # mat1 = mat1 * 0
    #
    # print(mat1)
    # print('######################################')
    # print(mat2)

    # url_to_gif('https://38.media.tumblr.com/3e7cbe14738b9335bd11faf698d3fc85/tumblr_np87p7jzfm1ux8f0go1_400.gif', '.')

    # msk = [11,4,33,0,1,22,27,25,16]
    # msk = np.array(msk).reshape([3,3,1])
    # from HandSeg.seg_colors import full_to_partial_seg
    # vec_partial_seg = np.vectorize(partial_seg)
    # msk = vec_partial_seg(msk, full_to_partial_seg)
    # print(msk.shape)
    # print(msk)

    img = Image.open("D:/private/datasets/handTrack/studies/full_hands/17.12.17/1/frames/0.jpg")
    # w, h = img.size
    # print(w, h)
    # print(img.mode)
    # a = image_to_mat(img, shape=(480, 640, 3))
    # print(a.shape)
    # im = mat_to_image(a, shape=(480, 640, 3))
    # im.show()
    img = img.convert('L').show()
    print(np.array(img).shape)