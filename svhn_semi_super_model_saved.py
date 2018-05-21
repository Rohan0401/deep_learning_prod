import os
import shutil

import tensorflow as tf

from gan import GAN

"""
Loads the saved GAN model, inject additional layers for the input transformation and export into protobuf format
"""

# Command line arguments

tf.app.flags.DEFINE_string ('checkpoints_dir', './checkpoints',
                            """Directory where to read training checkpoints""")
tf.app.flags.DEFINE_string ('output_dir', './gan-export',
                            """Directory where to export the model""")
tf.app.flags.DEFINE_integer ('model_version', 1,
                             """Version number of the model""")

FLAGS = tf.app.flags.FLAGS


def preprocess_image(image_buffer):
    """
    Preprocessed JPEG encoded bytes to 3D tensor and scales it to range of [-1 , 1]
    :param image_buffer: Input buffer to contain JPEG image
    :return: 4D image tensor (1, width, height, channel) with pixels scaled
                to [-1, 1]. First dimension is the batch size
    """

    # Decode the string as a RGB JPEG
    # Resulting image contains unknown height and width
    # that is set dynamically by decode_jpeg so it is unknown at compile time

    image = tf.image.decode_jpeg (image_buffer, channels=3)

    """
    At this point all the image pixel stays in [0, 1] until they scaled to [-1, 1].
    The various adjust_* ops all requires this range for dtype float
    """
    image = tf.image.convert_image_dtype (image, dtype=tf.float32)

    """
    The network accepts image in batches so we preserve the batch_size as first parameter.
    In this case the batch size is 1
    """
    image = tf.expand_dims (image, 0)

    # finally we scale the images to [-1, 1] from [0, 1]

    image = tf.subtract (image, 0.5)
    image = tf.multiply (image, 2.0)

    return image


def main(_):
    with tf.Graph ().as_default ():
        # Inject placeholders in graph
        serialized_tf_example = tf.placeholder (tf.string, name='input_image')

        feature_configs = {
            'image/encoded': tf.FixedLenFeature (
                shape=[], dtype=tf.string,
            )
        }
        tf_example = tf.parse_example (serialized_tf_example, feature_configs)
        jpegs = tf_example['image/encoded']
        images = tf.map_fn (preprocess_image, jpegs, dtype=tf.float32)
        images = tf.squeeze (images, [0])

        # now the images have been changed to [1, ? , ? , 3]

        # Create a GAN model
        z_size = 100
        learning_rate = 0.0003

        net = GAN (images, z_size, learning_rate, drop_rate=0.)

        # Create a saver to restore the model

        saver = tf.train.Saver ()

        with tf.Session () as sess:
            # Restore the model from last checkpoint
            ckpt = tf.train.get_checkpoint_state (FLAGS.checkpoints_dir)
            saver.restore (sess, ckpt.model_checkpoint_path)

            # (Re) create the export directory
            export_path = os.path.join (
                tf.compat.as_bytes (FLAGS.output_dir),
                tf.compat.as_bytes (str (FLAGS.model_version))
            )
            if os.path.exists (export_path):
                shutil.rmtree (export_path)

            # Create model builder

            builder = tf.saved_model.builder.SavedModelBuilder (export_path)

            # create tensor info
            predict_tensor_inputs_info = tf.saved_model.utils.build_tensor_info (jpegs)
            predict_tensor_scores_info = tf.saved_model.utils.build_tensor_info (net.discriminator_out)

            # predict signature of the methods

            predict_signature = (
                tf.saved_model.signature_def_utils.build_signature_def (
                    inputs={'images': predict_tensor_inputs_info},
                    outputs={'scores': predict_tensor_scores_info},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                )
            )

            # Save model

            legecy_init_op = tf.group (tf.tables_initializer (), name='legacy_init_op')
            builder.add_meta_graph_and_variables (
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'predict_images': predict_signature
                },

                legacy_init_op=legecy_init_op
            )

            builder.save ()

            print ("Sucessfully exported GAN model version '{}' into '{}'".format (
                FLAGS.model_version, FLAGS.output_dir
            ))


if __name__ == '__main__':
    tf.app.run ()
