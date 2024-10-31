from utils.image_processor import ImageProcessor
from PIL import Image
import torch


if __name__ == "__main__":
    image_path = "28e31bc5-0c7a-47a4-b6b4-f3a614f94284.jpg"
    prompt = "container"

    device = torch.device("cuda")
    processor = ImageProcessor(device=device)

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    input_image, detections, output_image = processor.process_image(Image.open(image_path), prompt)
    input_image.save("input.jpg")
    output_image.save("output.jpg")
    mask = Image.fromarray(detections[0].mask[0])
    mask.save("mask.jpg")