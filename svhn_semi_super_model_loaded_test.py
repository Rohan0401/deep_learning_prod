import tensorflow as tf
import numpy as np

import utils

"""
Loads the saved GAN model and predict of test images
"""


def load_test_images():
    """
    load random iamges from SVNH dataset and test with the model
    :return:
    """
    utils.download_train_test_data ()
    _, testset = utils.load_data_sets ()

    idx = np.random.randint (0, testset['X'].shape[3], size=64)
    test_images = testset['X'][:, :, :, idx]
    test_lables = testset['y'][idx]

    test_images = np.rollaxis (test_images, 3)
    test_images = utils.scale (test_images)

    return test_images, test_lables


def load_and_predict_with_checkpoint():
    """
    Loads save checkpoints and test the model performance
    :return:
    """
    # load test images and labels

    test_images, test_labels = load_test_images ()

    # Create an empty graph
    loaded_graph = tf.Graph ()

    with tf.Session (graph=loaded_graph) as sess:
        # restore saved graph
        saver = tf.train.import_meta_graph ('./checkpoints/generator.ckpt.meta')
        saver.restore (sess, tf.train.latest_checkpoint ('./checkpoints'))

        # Get tensor by name

        pred_class_tensor = loaded_graph.get_tensor_by_name ("pred_class:0")
        input_real_tensor = loaded_graph.get_tensor_by_name ("input_real:0")
        y_tensor = loaded_graph.get_tensor_by_name ("y:0")
        drop_rate_tensor = loaded_graph.get_tensor_by_name ("drop_rate:0")
        correct_pred_sum_tensor = loaded_graph.get_tensor_by_name ("correct_pred_sum:0")

        # make prediction

        correct, pred_class = sess.run ([correct_pred_sum_tensor, pred_class_tensor],
                                        feed_dict={
                                            input_real_tensor: test_images,
                                            y_tensor: test_labels,
                                            drop_rate_tensor: 0.
                                        })
        # Print results
        print ("No. of correct predictions : {}".format (correct))
        print ("Predicted classes: {}".format (pred_class))


def load_and_predict_with_saved_model():
    """
    Loads saved as protobuf model and make prdiction on a single image
    :return:
    """
    with tf.Session (graph=tf.Graph ()) as sess:
        export_dir = './gan-export/1'
        # restore the model
        model = tf.saved_model.loader.load (sess, [tf.saved_model.tag_constants.SERVING], export_dir)

        # print(model)
        loaded_graph = tf.get_default_graph ()

        # get tensor by name

        input_tensor_name = model.signature_def['predict_images'].inputs['images'].name
        input_tensor = loaded_graph.get_tensor_by_name (input_tensor_name)
        output_tensor_name = model.signature_def['predict_images'].outputs['scores'].name
        output_tensor = loaded_graph.get_tensor_by_name (output_tensor_name)

        # make prediction
        image_file_name = './svhn_test_images/image_3.jpg'

        with open (image_file_name, 'rb') as f:
            image = f.read ()
            scores = sess.run (output_tensor, {input_tensor: [image]})

        # print results
        print ("Image file name : {}".format (image_file_name))
        print ("Scores: {}".format (scores))


def main(_):
    load_and_predict_with_checkpoint ()
    load_and_predict_with_saved_model ()


if __name__ == '__main__':
    tf.app.run ()