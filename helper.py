import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import cv2


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    hist_eq_flag = True #flag to perform histogram eq on input data
    image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
    label_paths = {
        re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
        for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
    background_color = np.array([255, 0, 0])

    random.shuffle(image_paths)
    # perform an 80-20 train -validation split
    num_samples = len(image_paths)
    num_train = int(0.8 * num_samples)
    num_valid = num_samples - num_train
    train_image_paths = image_paths[0:num_train]
    valid_image_paths = image_paths[num_train + 1:]
    print("Num train samples, valid samples ", num_train, num_valid)

    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        for batch_i in range(0, len(train_image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in train_image_paths[batch_i:batch_i + batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]
                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
                images.append(image)
                gt_images.append(gt_image)
            # augment data
            images_arr = np.array(images)
            if hist_eq_flag:
                images_arr = hist_eq(images_arr)
            yield np.array(images_arr), np.array(gt_images)

    def get_validation_batches_fn(batch_size):
        """
        Create batches of validation data
        :param batch_size: Batch Size
        :return: Batches of validation data
        """
        for batch_i in range(0, len(valid_image_paths), batch_size):
            valid_images = []
            valid_gt_images = []
            for image_file in valid_image_paths[batch_i:batch_i + batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]
                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
                valid_images.append(image)
                valid_gt_images.append(gt_image)
            valid_images_arr = np.array(valid_images)
            if hist_eq_flag:
                valid_images_arr = hist_eq(valid_images_arr)
            yield np.array(valid_images_arr), np.array(valid_gt_images)

    return get_batches_fn, get_validation_batches_fn


def gen_validation_data(valid_image_paths, label_paths, image_shape):
    background_color = np.array([255, 0, 0])

    valid_images, valid_gt_images = [], []
    for image_file in valid_image_paths:
        gt_image_file = label_paths[os.path.basename(image_file)]
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
        gt_bg = np.all(gt_image == background_color, axis=2)
        gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
        gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
        valid_images.append(image)
        valid_gt_images.append(gt_image)

    return np.array(valid_images), np.array(valid_gt_images)


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    hist_eq_flag = True
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        images = [image]
        images_arr = np.array(images)
        if hist_eq_flag:
            images_arr = hist_eq(images_arr)
        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [images_arr[0,...]]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

        
def compute_train_statistics(image_data_train):
    """Returns
        min: min val of training samples
        max: max val of training samples"""
    X_min, X_max = np.min(image_data_train), np.max(image_data_train)
    return (X_min, X_max)

        
def min_max_scaling(image_data, X_min, X_max):
    a, b = -1.0, 1.0
    image_data[:, ...] = a + ((image_data - X_min) * (b - a)) / (X_max - X_min)
    return image_data

def hist_eq(image_data):
    image_data_out = np.zeros_like(image_data)
    for i in range(len(image_data)):
        img = image_data[i,...]#np.reshape(image_data[i,...], (32, 32, 3))
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        y, cr, cb = cv2.split(img_ycrcb)
        y = cv2.equalizeHist(y)
        img_ycrcb = cv2.merge((y, cr, cb))
        img = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCR_CB2BGR)
        image_data_out[i,...] = img#np.reshape(img,(1,32*32*3))
    return image_data_out



def gen_video_output(sess, logits, keep_prob, image_pl, frame, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param frame: video frame
    :param image_shape: Tuple - Shape of image
    :return: Output processed frame
    """
    # image = frame
    image = scipy.misc.imresize(frame, image_shape)
    # print("image shape and expected shape in helper is ", image.shape, image_shape)

    im_softmax = sess.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 1.0, image_pl: [image]})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)
    street_im_arr = np.array(street_im)
    # print("street im shape is", street_im_arr.shape)
    return street_im_arr












