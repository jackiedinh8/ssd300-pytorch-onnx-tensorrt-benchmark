import cv2
import numpy as np
import onnx
import onnxoptimizer
import onnxruntime as rt
import tensorrt as trt
import time
import torch
import torchvision.transforms as transforms

from utils import draw_bboxes

class SSDOnnx:
    def __init__(self, path):
        self.path = path
        self.session = rt.InferenceSession(path)

    def do_infer(self, image_path):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed_image = transform(image)
        tensor = torch.tensor(transformed_image, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).numpy()

        input_name = self.session.get_inputs()[0].name
        label_name1 = self.session.get_outputs()[0].name
        label_name2 = self.session.get_outputs()[1].name
        locs, labels = self.session.run([label_name1, label_name2], {input_name: tensor})
        locs_tensor = torch.Tensor(locs)
        labels_tensor = torch.Tensor(labels)

        results_per_input = utils.decode_results((locs_tensor, labels_tensor))
        best_results_per_input = [utils.pick_best(results, 0.45) for results in results_per_input]
        classes_to_labels = utils.get_coco_object_dictionary()
        image_result = draw_bboxes(image, best_results_per_input, classes_to_labels)

        # save the image to disk
        save_name = "onnx_" + image_path.split('/')[-1]
        print('saving file: {}'.format(save_name))
        cv2.imwrite(f"outputs/{save_name}", image_result)

    def optimize(self, model_path):
        test_model = onnx.load(self.path)
        optimizers_list = onnxoptimizer.get_fuse_and_elimination_passes()
        optimized_model = onnxoptimizer.optimize(test_model, optimizers_list, fixed_point=True)
        onnx.checker.check_model(optimized_model)
        with open(model_path, "wb") as f:
                f.write(optimized_model.SerializeToString())

    def do_benchmark(self, batch_size=1):
        input_name = self.session.get_inputs()[0].name
        label_name = self.session.get_outputs()[0].name
        x = torch.ones((batch_size, 3, 300, 300)).numpy()

        print('ssd300 onnx benchmarking with batch size {} ...'.format(batch_size))
        start = time.time()
        for i in range(1000):
            pred = self.session.run([label_name], {input_name: x})[0]
        end = time.time()
        fps = batch_size * 1000 / (end - start)
        print('fps: {}'.format(fps))

    def export_tensorrt(self, model_path):
        G_LOGGER = trt.Logger(trt.Logger.WARNING)
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        trt.init_libnvinfer_plugins(G_LOGGER, '')

        with trt.Builder(G_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, \
                trt.OnnxParser(network, G_LOGGER) as parser:
            builder.max_batch_size = 16
            builder.max_workspace_size = 1 << 30
            builder.fp16_mode = True
            with open(self.path, 'rb') as model:
                if not parser.parse(model.read()):
                    for e in range(parser.num_errors):
                        print(parser.get_error(e))
                    raise TypeError("Parsring onnx model failed")
            engine = builder.build_cuda_engine(network)
            with open(model_path, 'wb') as f:
                f.write(engine.serialize())

