import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
#tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # 1x1 convolution layer
    layer7_conv_1x1_out = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1, 1), padding='same',
                                           kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    layer4_conv_1x1_out = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides=(1, 1), padding='same',
                                           kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    layer3_conv_1x1_out = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides=(1, 1), padding='same',
                                           kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    print("conv1x1 shape is", layer7_conv_1x1_out.get_shape())
    # transpose convolutions
    conv2d_transp_out1 = tf.layers.conv2d_transpose(layer7_conv_1x1_out, num_classes, 4, strides=(2, 2), padding='same',
                                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    skip_out1 = tf.add(conv2d_transp_out1, layer4_conv_1x1_out)
    conv2d_transp_out2 = tf.layers.conv2d_transpose(skip_out1, num_classes, 4, strides=(2, 2), padding='same',
                                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    skip_out2 = tf.add(conv2d_transp_out2, layer3_conv_1x1_out)
    conv2d_transp_out3 = tf.layers.conv2d_transpose(skip_out2, num_classes, 16, strides=(8, 8), padding='same',
                                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    return conv2d_transp_out3
#tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)
    prediction = tf.argmax(tf.reshape(tf.nn.softmax(logits), tf.shape(nn_last_layer)), dimension=3)
    c_label = tf.argmax(correct_label, dimension=3)
    iou, iou_op = tf.metrics.mean_iou(c_label, prediction, num_classes)
    return logits, train_op, cross_entropy_loss, iou, iou_op
#tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, get_validation_batches_fn, train_op, cross_entropy_loss,
             iou, iou_op, input_image, correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    first_flag = True
    for epoch in range(epochs):
        print("EPOCH {} ...".format(epoch + 1))
        for batch_x, batch_y in get_batches_fn(batch_size):
            if first_flag:
                print("batch x, y shape is", batch_x.shape, batch_y.shape)
                first_flag = False
            print("*", end="")
            feed_dict = {input_image: batch_x, correct_label: batch_y, learning_rate: 0.001, keep_prob: 0.35}
            sess.run(train_op, feed_dict)
        print("[DONE]")
        loss = sess.run(cross_entropy_loss, feed_dict)
        print("Training loss is ", loss)
        sess.run(iou_op, feed_dict)
        print("Training mean IoU is", sess.run(iou, feed_dict))
        # model validation
        valid_loss, valid_mean_iou = 0.0, 0.0
        for i in range(5):
            batch_x, batch_y = next(get_validation_batches_fn(batch_size))
            if i == 0:
                print("validation x,y shape is", batch_x.shape, batch_y.shape)
            feed_dict = {input_image: batch_x, correct_label: batch_y, learning_rate: 0.001, keep_prob: 1.0}
            valid_loss += sess.run(cross_entropy_loss, feed_dict)
            sess.run(iou_op, feed_dict)
            valid_mean_iou += sess.run(iou, feed_dict)

        print("Validation loss is ", valid_loss / 5.)
        print("Validation mean IoU is ", valid_mean_iou / 5.)
#tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)  # (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    epochs = 3  # 35
    batch_size = 10

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn, get_validation_batches_fn = helper.gen_batch_function(
            os.path.join(data_dir, 'data_road/training'), image_shape)

        correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
        learning_rate = tf.placeholder(tf.float32)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, train_op, cross_entropy_loss, iou, iou_op = optimize(nn_last_layer, correct_label, learning_rate,
                                                                     num_classes)
        # Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        # ajay - Required for mean iou computation
        sess.run(tf.local_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, get_validation_batches_fn, train_op,
                 cross_entropy_loss, iou, iou_op, image_input, correct_label, keep_prob, learning_rate)

        # save the model
        saver = tf.train.Saver()
        save_path = saver.save(sess, "./models/model.ckpt")
        print("Model saved in file: %s" % save_path)

def infer():
    data_dir = './data'
    runs_dir = './runs'
    image_shape = (160, 576)
    num_classes = 2
    # load saved model
    tf.reset_default_graph()

    with tf.Session() as sess:
        # Restore variables from disk.
        # This is painful in tensor flow. Recreate the model to restore it
        correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
        learning_rate = tf.placeholder(tf.float32)
        keep_prob = tf.placeholder(tf.float32)
        vgg_path = os.path.join(data_dir, 'vgg')
        image_input, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, train_op, cross_entropy_loss, iou, iou_op = optimize(nn_last_layer, correct_label, learning_rate,
                                                                     num_classes)
        saver = tf.train.Saver()
        saver.restore(sess, "./models/model.ckpt")
        print("Model restored.")
        #Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

if __name__ == '__main__':
    run()
    infer()

