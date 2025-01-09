import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
from PIL import Image
import numpy as np
import io
from transformers import AutoProcessor, AutoModelForCausalLM  

import argparse
import os

from streaming import MDSWriter
from composer.utils import dist, get_device
import ocifs
from datasets import load_dataset
import pickle
import json
import cv2
from tqdm import tqdm

from transformers import CLIPProcessor

fs = ocifs.OCIFileSystem(config = '/secrets/oci/config')
remote = "oci://mosaicml-internal-datasets/mosaicml-internal-dataset-multi-image/synthetic-compo" 
columns = {
    'images': 'bytes',
    'messages': 'json',
}

base_dir = "mosaicml-internal-datasets@axhe5a72vzpp/mosaicml-internal-dataset-multi-image/abo-raw/abo-spins/spins" 

compos_dirs = ['left', 'right', 'above', 'below']

def run_example(model, processor, device_id, image_ref, task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image_ref, return_tensors="pt").to(device_id, torch.float16)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].to(device_id),
      pixel_values=inputs["pixel_values"].to(device_id),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    return parsed_answer

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--width", type=int, default=512, help="The width for generated image"
    )
    parser.add_argument(
        "--height", type=int, default=512, help="The height for generated image"
    )
    parser.add_argument(
        "--num_steps", type=int, default=25, help="The num_steps for diffusion process"
    )
    parser.add_argument(
        "--controlnet_scale", type=float, default=0.7, help="The guidance for diffusion process"
    )
    parser.add_argument(
        "--seed", type=int, default=123456789, help="A seed for reproducible inference"
    )
    parser.add_argument(
        "--true_gs", type=float, default=3.5, help="true guidance"
    )
    parser.add_argument(
        "--num_generations", type=int, default = 50000, help="number of images to generate"
    )
    parser.add_argument(
        "--aesthetic_model_id", type=str, default="shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE", help="aesthetic predictor"
    )
    parser.add_argument(
        "--dist_timeout", type=float, default=300.0, help="dist timeout"
    )
    return parser

def main(args, writer):
    device_id = f'cuda:{dist.get_local_rank()}'
    print('loading controlnet on rank 0')
    if dist.get_local_rank() == 0:
        controlnet = FluxControlNetModel.from_pretrained(
            "Xlabs-AI/flux-controlnet-canny-diffusers",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
        print('loading pipe on rank 0')
        pipe = FluxControlNetPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            controlnet=controlnet,
            torch_dtype=torch.bfloat16
        ).to(device_id)
    print('finished downloading')
    dist.barrier()
    print('loading controlnet on non-rank 0')
    if dist.get_local_rank() != 0:
        controlnet = FluxControlNetModel.from_pretrained(
            "Xlabs-AI/flux-controlnet-canny-diffusers",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
        print('loading pipe on non-rank 0')
        pipe = FluxControlNetPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            controlnet=controlnet,
            torch_dtype=torch.bfloat16
        ).to(device_id)
    print('finish downloading on non-rank 0')
    dist.barrier()

    print('loading florence on rank 0')
    if dist.get_local_rank() == 0:
        florence_processor = AutoProcessor.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True)
        florence_model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True, torch_dtype='auto').eval().to(device_id)
    print('finish loading florence on rank 0')
    dist.barrier()
    print('loading florence on non-rank 0')
    if dist.get_local_rank() != 0:
        florence_processor = AutoProcessor.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True)
        florence_model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True, torch_dtype='auto').eval().to(device_id)
    print('finish loading florence on non-rank 0')
    dist.barrier() 

    print('get all spins')
    all_spins = []
    with fs.open(base_dir + "/all_spins.json", 'rb') as jsonfile:
        for line in jsonfile:
            all_spins.append(json.loads(line))

    samples_per_rank, remainder = divmod(args.num_generations, dist.get_world_size())

    start_idx = dist.get_global_rank() * samples_per_rank + min(remainder, dist.get_global_rank())
    end_idx = start_idx + samples_per_rank
    if dist.get_global_rank() < remainder:
        end_idx += 1

    print('index all images')
    images = fs.glob(f'{base_dir}/original/*.jpg')

    for _ in tqdm(range(start_idx, end_idx)):
        # control_image = load_image("https://huggingface.co/Xlabs-AI/flux-controlnet-canny-diffusers/resolve/main/canny_example.png")
        # prompt = "handsome girl with rainbow hair, anime"
        
        compos_images_indices = np.random.choice(len(all_spins), 2, replace = False)
        compos_json1 = all_spins[compos_images_indices[0]]
        compos_json2 = all_spins[compos_images_indices[1]]

        compos_name1 = compos_json1["product_type"][0]["value"].replace("_", " ").lower()
        compos_jpegname1 = f'{base_dir}/original/{compos_json1["spin_id"]}_00.jpg'
        compos_name2 = compos_json2["product_type"][0]["value"].replace("_", " ").lower()
        compos_jpegname2 = f'{base_dir}/original/{compos_json2["spin_id"]}_00.jpg'

        if compos_jpegname1 not in images or compos_jpegname2 not in images:
            continue

        
        with fs.open(compos_jpegname1) as f:
            image1 = Image.open(io.BytesIO(f.read())).convert("RGB")
            image1.thumbnail((args.width//2, args.height//2), Image.Resampling.LANCZOS)


        with fs.open(compos_jpegname2) as f:
            image2 = Image.open(io.BytesIO(f.read())).convert("RGB")
            image2.thumbnail((args.width//2, args.height//2), Image.Resampling.LANCZOS)

        
        opencv_image1 = cv2.cvtColor(np.array(image1, dtype = np.uint8), cv2.COLOR_RGB2GRAY)
        opencv_image2 = cv2.cvtColor(np.array(image2, dtype = np.uint8), cv2.COLOR_RGB2GRAY)

        edges1 = cv2.Canny(opencv_image1, 50, 200)
        edges2 = cv2.Canny(opencv_image2, 50, 200)

        canny_image_final = np.zeros((args.height, args.width), dtype = np.uint8)
        
        compos_dir_chosen = np.random.choice(4, 2, replace = False)
        correct_compos_dir = compos_dirs[compos_dir_chosen[0]]
        wrong_compos_dir = compos_dirs[compos_dir_chosen[1]]

        prompt = f'{compos_name1} {correct_compos_dir} of {compos_name2}'
        wrong_prompt = f'{compos_name1} {wrong_compos_dir} of {compos_name2}'

        if correct_compos_dir == 'left':
            canny_image_final[args.height//4: args.height//4 + edges1.shape[0], :edges1.shape[1]] = edges1
            canny_image_final[args.height//4: args.height//4 + edges2.shape[0], -edges2.shape[1]:] = edges2
        elif correct_compos_dir == 'right':
            canny_image_final[args.height//4: args.height//4 + edges1.shape[0], -edges1.shape[1]:] = edges1
            canny_image_final[args.height//4: args.height//4 + edges2.shape[0], :edges2.shape[1]] = edges2
        elif correct_compos_dir == 'below':
            canny_image_final[-edges1.shape[0]:, args.width//4: args.width//4 + edges1.shape[1]] = edges1
            canny_image_final[:edges2.shape[0], args.width//4: args.width//4 + edges2.shape[1]] = edges2
        else:
            canny_image_final[:edges1.shape[0], args.width//4: args.width//4 + edges1.shape[1]] = edges1
            canny_image_final[-edges2.shape[0]:, args.width//4: args.width//4 + edges2.shape[1]] = edges2
        
        control_image = Image.fromarray(cv2.cvtColor(canny_image_final, cv2.COLOR_GRAY2RGB))


        print(prompt)

        #prompt = f'{prompt_dataset[sample_id]['caption1']} in the style of {args.lora_type}'
        image = pipe(
            prompt,
            control_image=control_image,
            controlnet_conditioning_scale=args.controlnet_scale,
            num_inference_steps=args.num_steps,
            guidance_scale=args.true_gs,
            height=args.height,
            width=args.width,
            num_images_per_prompt=1,
        ).images[0]
        print(wrong_prompt)

        output_imgs = pickle.dumps([image.convert("RGB")])


        rating = run_example(florence_model, florence_processor, device_id, image, '<OPEN_VOCABULARY_DETECTION>', text_input=f'{compos_name1}, {compos_name2}')
        labels = rating['<OPEN_VOCABULARY_DETECTION>'].get('bboxes_labels', [])
        if len(labels) == 0:
            print("rejected")
            continue

        messages = []

        if np.random.rand() > 0.5:
            messages.append({'role': 'user', 'content': f'<image>\n How does this image with the prompt "{prompt}" match the desired compositionality condition of {correct_compos_dir}?'})
            messages.append({'role': 'assistant', 'content': f'This correctly models the compositionality direction {correct_compos_dir}.'})
        else:
            messages.append({'role': 'user', 'content': f'<image>\n How does this image with the prompt "{wrong_prompt}" match the desired compositionality condition of {wrong_compos_dir}?'})
            messages.append({'role': 'assistant', 'content': f'This models the compositionality direction {correct_compos_dir} instead of the desired {wrong_compos_dir}.'})

        writer.write({'images': output_imgs, 'messages': messages})

        # if not os.path.exists(args.save_path):
        #     os.mkdir(args.save_path)
        # ind = len(os.listdir(args.save_path))
        # result.save(os.path.join(args.save_path, f"result_{ind}.png"))
        args.seed = args.seed + 1


if __name__ == "__main__":
    print("parse arguments")
    args = create_argparser().parse_args()
    print('get device')
    device = get_device()
    # print('i love duyiqing')
    print('initializing dist')
    dist.initialize_dist(device, args.dist_timeout)
    print('making writer')
    writer = MDSWriter(out = f'{remote}/compos-rank{dist.get_global_rank()}', compression = "zstd", columns = columns)
    main(args, writer)
    writer.finish()