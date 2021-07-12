import argparse

from ssd import SSD300
from ssd_onnx import SSDOnnx
from ssd_tensorrt import SSDTensorRT

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', required=True, help='path to the input data'
    )
    args = vars(parser.parse_args())
    image_path = args['input']

    # load ssd300 model, do inference on sample image and run benchmark
    ssd_model = SSD300()
    ssd_model.do_infer(image_path)
    ssd_model.do_benchmark(1)
    ssd_model.do_benchmark(8)
    #ssd_model.do_benchmark(16)
    #ssd_model.save_model('models/model.pt')
    ssd_model.export_onnx('models/model.onnx')

    # load ssd300 onnx model, do inference on sample image and run benchmark
    onnx_model = SSDOnnx('models/model.onnx')
    onnx_model.do_infer(image_path)
    onnx_model.do_benchmark(1)
    onnx_model.optimize('models/op_model.onnx')
    onnx_model.export_tensorrt('models/model.engine')

    #op_onnx_model = SSDOnnx('models/op_model.onnx')
    ##op_onnx_model.do_infer(image_path)
    ##op_onnx_model.do_benchmark(1)
    #op_onnx_model.export_tensorrt('models/model.engine')

    tensorrt_model = SSDTensorRT('models/model.engine')
    tensorrt_model.do_infer(image_path)
    tensorrt_model.do_benchmark(1)
    tensorrt_model.do_benchmark(8)
