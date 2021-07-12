import cv2
import time
import torch
import torchvision.transforms as transforms

from utils import draw_bboxes

class SSD300:
    def __init__(self):
        self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')

    def do_infer(self, image_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()
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
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            detections = self.model(tensor)
            results_per_input = utils.decode_results(detections)
            best_results_per_input = [utils.pick_best(results, 0.45) for results in results_per_input]
            classes_to_labels = utils.get_coco_object_dictionary()
            image_result = draw_bboxes(image, best_results_per_input, classes_to_labels)
            # save the image to disk
            save_name = 'ssd300_' + image_path.split('/')[-1]
            print('saving file: {}'.format(save_name))
            cv2.imwrite(f"outputs/{save_name}", image_result)

    def do_benchmark(self, batch_size=1):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = torch.ones((batch_size, 3, 300, 300)).to(device)
        print('ssd300 benchmarking with batch size {} ...'.format(batch_size))
        start = time.time()
        with torch.no_grad():
            for i in range(1000):
                y = self.model(x)
        end = time.time()
        fps = batch_size * 1000 / (end - start)
        print('fps: {}'.format(fps))

    def save_model(self, path):
        utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()

        input = torch.ones((1, 3, 300, 300)).to(device)
        output = self.model.forward(input)

        traced_model = torch.jit.trace(self.model, input)
        traced_model.save(path)


    def export_onnx(self, path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = torch.ones((1, 3, 300, 300)).to(device)
        y = self.model(x)

        _ = torch.onnx.export(self.model, x, path,
                              example_outputs=y,
                              training=False,
                              export_params=True,
                              input_names=['input0'],
                              output_names=['output0', 'output1'])


