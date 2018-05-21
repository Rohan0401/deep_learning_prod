"""
This sends the JPEG image to tensorflow_model_server loaded with GAN model.

The code has been complied together with TensorFlow serving, not locally. The client is TensorFlow Docker container

"""

import time

from argparse import ArgumentParser

# Connect TensorFLow via gRPC

from grpc.beta import implementations
import tensorflow as tf


# TensorFlow Serving to send messages

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.contrib.util import make_tensor_proto

def parse_arg():
    parser = ArgumentParser(description="Request a TensorFlow server for a prediction on the image")
    parser.add_argument("-s", "--saver",
                        dest="server",
                        default="172.17.0.2:9000",
                        help="prediction service host:port")
    parser.add_argument("-i", "--image",
                        dest="image",
                        default="",
                        help="path to image in JPEG format",)
    args = parser.parse_args()

    host, port = args.server.split(":")

    return host, port, args.image

def main():

    # parse command line arguments

    host, port, image = parse_arg()

    chennel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(chennel)

    # Send request

    with open(image, 'rb') as f:
        # See prediction_service.proto for gRPC request/response details.

        data = f.read()

        start = time.time()

        request = prediction_service_pb2()

        # Call GAN model to make prediction on the image

        # Call GAN model to make prediction on the image

        request.model_spec.name = 'gan'
        request.model_spec.signature_name = 'predict_images'
        request.inputs['images'].CopyFrom(make_tensor_proto(data, shape=[1]))

        result = stub.Predict(request, 60.0) # 60 second timeout

        end = time.time()
        time_diff = end - start

        print(result)
        print("Time eplapsed: {}".format(time_diff))

    if __name__ == '__main__':
        main()

