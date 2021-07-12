import cv2
import numpy as np
import time
import torch
import torchvision.transforms as transforms
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from utils import draw_bboxes

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine, batch_size=1):
    """Allocates host and device buffer for TRT engine inference.

    This function is similair to the one in common.py, but
    converts network outputs (which are np.float32) appropriately
    before writing them to Python buffer. This is needed, since
    TensorRT plugins doesn't support output type description, and
    in our particular case, we use NMS plugin as network output.

    Args:
        engine (trt.ICudaEngine): TensorRT engine

    Returns:
        inputs [HostDeviceMem]: engine input memory
        outputs [HostDeviceMem]: engine output memory
        bindings [int]: buffer to device bindings
        stream (cuda.Stream): cuda stream for engine inference synchronization
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    # Current NMS implementation in TRT only supports DataType.FLOAT but
    # it may change in the future, which could brake this sample here
    # when using lower precision [e.g. NMS output would not be np.float32
    # anymore, even though this is assumed in binding_to_type]
    binding_to_type = {"input0": np.float32, "output0": np.float32, "output1": np.int32}

    for binding in engine:
        print('binding input "{}" to type'.format(binding))
        #size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dtype = binding_to_type[str(binding)]
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


class SSDTensorRT:
    def __init__(self, path):
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.TRT_LOGGER, '')
        self.runtime = trt.Runtime(self.TRT_LOGGER)
        with open(path, 'rb') as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)

    # This function is generalized for multiple inputs/outputs.
    # inputs and outputs are expected to be lists of HostDeviceMem objects.
    def do_infer_internal(self, context, bindings, inputs, outputs, stream, batch_size=1):
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        stream.synchronize()
        return [out.host for out in outputs]

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

        inputs, outputs, bindings, stream = allocate_buffers(self.engine)
        np.copyto(inputs[0].host, tensor.ravel())
        context = self.engine.create_execution_context()
        [locs_out, labels_out] = self.do_infer_internal(
            context, bindings=bindings, inputs=inputs,
            outputs=outputs, stream=stream)

        locs = np.reshape(locs_out, (1, 4, 8732))
        labels = np.reshape(labels_out, (1, 81, 8732))
        locs_tensor = torch.Tensor(locs)
        labels_tensor = torch.Tensor(labels)
        results_per_input = utils.decode_results((locs_tensor, labels_tensor))
        best_results_per_input = [utils.pick_best(results, 0.45) for results in results_per_input]
        classes_to_labels = utils.get_coco_object_dictionary()
        image_result = draw_bboxes(image, best_results_per_input, classes_to_labels)
        # save the image to disk
        save_name = "tensorrt_" + image_path.split('/')[-1]
        print('saving file: {}'.format(save_name))
        cv2.imwrite(f"outputs/{save_name}", image_result)

    def do_benchmark(self, batch_size=1):
        inputs, outputs, bindings, stream = allocate_buffers(self.engine, batch_size)

        x = np.ones((batch_size, 3, 300, 300))
        np.copyto(inputs[0].host, x.ravel())

        print('ssd tensorrt benchmarking with batch size {} ...'.format(batch_size))
        start = time.time()
        context = self.engine.create_execution_context()
        for i in range(1000):
            [detection_out, keepCount_out] = self.do_infer_internal(
                context, bindings=bindings, inputs=inputs,
                outputs=outputs, stream=stream)
        end = time.time()
        fps = 8000 / (end - start)
        print('fps: {}'.format(fps))
