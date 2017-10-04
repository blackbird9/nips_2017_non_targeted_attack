"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image
import os

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 128, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath) as f:
            image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            Image.fromarray(img).save(f, format='PNG')
            

def myclip(mytf, eps, source):
    max1 = tf.maximum(-1., tf.subtract(source, eps))
    max2 = tf.maximum(max1, mytf)
    min1 = tf.minimum(1., tf.add(source, eps))
    min2 = tf.minimum(min1, max2)
    return min2            


def main(_):
  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001
    checkpoint_path = FLAGS.checkpoint_path
    tensorflow_master = FLAGS.master

    tf.logging.set_verbosity(tf.logging.INFO)

    for filenames, images in load_images(FLAGS.input_dir, batch_shape):
        tf.reset_default_graph()
        with tf.Graph().as_default():
 
            X = tf.placeholder(tf.float32, shape=batch_shape)
            source = tf.placeholder(tf.float32, shape=batch_shape)
            ori_copy = images
            lazy_image = images

            with slim.arg_scope(inception.inception_v3_arg_scope()):
                _, end_points = inception.inception_v3(X, num_classes=num_classes, is_training=False,reuse=False)

            output = end_points['Predictions']
            
            preds_max = tf.reduce_max(output, 1, keep_dims=True)
            y_max = tf.to_float(tf.equal(output, preds_max))
            y_max = y_max / tf.reduce_sum(y_max, 1, keep_dims=True)
            max_loss = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y_max)

            for _ in range(25):

                max_grad, = tf.gradients(max_loss, X)
                signed_grad = tf.sign(max_grad)

                adv_x = tf.stop_gradient(X + 0.3*eps*signed_grad)
                adv_x = myclip(adv_x, eps, source)

                saver = tf.train.Saver(slim.get_model_variables())
                session_creator = tf.train.ChiefSessionCreator(
                          scaffold=tf.train.Scaffold(saver=saver),
                          checkpoint_filename_with_path=checkpoint_path,
                          master=tensorflow_master)
                with tf.train.MonitoredSession(session_creator=session_creator) as sess:
                    lazy_image = sess.run(adv_x, feed_dict={X:lazy_image, source:ori_copy})
    

            save_images(lazy_image, filenames, FLAGS.output_dir)


if __name__ == '__main__':
    tf.app.run()