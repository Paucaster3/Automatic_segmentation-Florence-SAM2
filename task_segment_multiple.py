from tqdm import tqdm
from utils.image_processor import ImageProcessor
from PIL import Image
import torch
import os
from argparse import ArgumentParser
from clearml import Task, Logger

task = Task.init(project_name="Container Segmentation", task_name="Segmentation - Florence + SAM")

parser = ArgumentParser()
parser.add_argument("--input_dir", type=str, default="input", help="Directory containing input images")
parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output masks")
parser.add_argument("--prompt", type=str, default="container", help="Prompt for the model")
parser.add_argument("--report_images", type=bool, default=False, help="Report images to ClearML")

if __name__ == "__main__":
    
    logger = Logger.current_logger()
    
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    prompt = args.prompt
    report_images = args.report_images

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    device = torch.device("cuda")
    processor = ImageProcessor(device=device)

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    for root, _, files in os.walk(input_dir):
        for image_name in tqdm(files):
            if image_name.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(root, image_name)

                input_image, detections, output_image = processor.process_image(Image.open(image_path), prompt)

                output_image.save(os.path.join(output_dir, f"output_{image_name}"))
                mask = Image.fromarray(detections[0].mask[0])
                mask.save(os.path.join(output_dir, f"mask_{image_name}"))
                
                logger.report_image(f"Output Image - {image_name}", "Output Image", image=output_image)
            
            