import argparse
from PIL import Image
import os

from src.flux.xflux_pipeline import XFluxPipeline
from composer.utils import dist, get_device
import ocifs
from datasets import load_dataset
import pickle

from aesthetics_predictor import AestheticsPredictorV2Linear
from transformers import CLIPProcessor

fs = ocifs.OCIFileSystem(config = '/secrets/oci/config')
remote = "oci://mosaicml-internal-datasets/mosaicml-internal-dataset-multi-image/synthetic-aesthetic" 
columns = {
    'images': 'bytes',
    'messages': 'json',
}


def create_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt", type=str, required=True,
        help="The input text prompt"
    )
    parser.add_argument(
        "--neg_prompt", type=str, default="",
        help="The input text negative prompt"
    )
    parser.add_argument(
        "--local_path", type=str, default=None,
        help="Local path to the model checkpoint (Controlnet)"
    )
    parser.add_argument(
        "--repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (Controlnet)"
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="A filename to download from HuggingFace"
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=1,
        help="The number of images to generate per prompt"
    )

    parser.add_argument(
        "--model_type", type=str, default="flux-dev",
        choices=("flux-dev", "flux-dev-fp8", "flux-schnell"),
        help="Model type to use (flux-dev, flux-dev-fp8, flux-schnell)"
    )
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
        "--guidance", type=float, default=4, help="The guidance for diffusion process"
    )
    parser.add_argument(
        "--seed", type=int, default=123456789, help="A seed for reproducible inference"
    )
    parser.add_argument(
        "--true_gs", type=float, default=3.5, help="true guidance"
    )
    parser.add_argument(
        "--timestep_to_start_cfg", type=int, default=5, help="timestep to start true guidance"
    )
    parser.add_argument(
        "--save_path", type=str, default='results', help="Path to save"
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
    image = None
    device_id = f'cuda:{dist.get_local_rank()}' 

    if dist.get_local_rank() == 0:
        xflux_pipeline = XFluxPipeline(args.model_type, device_id)
    dist.barrier()
    if dist.get_local_rank() != 0:
        xflux_pipeline = XFluxPipeline(args.model_type, device_id)


    if dist.get_local_rank() == 0:
        prompt_dataset = load_dataset("sentence-transformers/coco-captions", split = "train")
    dist.barrier()
    if dist.get_local_rank() != 0:
        prompt_dataset = load_dataset("sentence-transformers/coco-captions", split = "train")

    dist.barrier()

    if dist.get_local_rank() == 0:
        processor = CLIPProcessor.from_pretrained(args.aesthetic_model_id)
        aesthetic_model = AestheticsPredictorV2Linear.from_pretrained(args.aesthetic_model_id).to(device_id)
    
    dist.barrier()

    if dist.get_local_rank() != 0:
        processor = CLIPProcessor.from_pretrained(args.aesthetic_model_id)
        aesthetic_model = AestheticsPredictorV2Linear.from_pretrained(args.aesthetic_model_id).to(device_id)
    
    dist.barrier()


    samples_per_rank, remainder = divmod(args.num_generations, dist.get_world_size())

    start_idx = dist.get_global_rank() * samples_per_rank + min(remainder, dist.get_global_rank())
    end_idx = start_idx + samples_per_rank
    if dist.get_global_rank() < remainder:
        end_idx += 1



    for sample_id in tqdm(range(start_idx, end_idx)):
        prompt = f'{prompt_dataset[sample_id]['caption1']}, aesthetic'
        result = xflux_pipeline(
            prompt=prompt,
            controlnet_image=image,
            width=args.width,
            height=args.height,
            guidance=args.guidance,
            num_steps=args.num_steps,
            seed=args.seed,
            true_gs=args.true_gs,
            neg_prompt=args.neg_prompt,
            timestep_to_start_cfg=args.timestep_to_start_cfg,
        )
        output_imgs = pickle.dumps([result.convert("RGB")])

        aesthetic_inputs = {k: v.to(device_id) for k, v in processor(images=result.convert("RGB"), return_tensors="pt").items()}

        with torch.no_grad():
            prediction = aesthetic_model(**inputs).logits




        messages = []
        messages.append({'role': 'user', 'content': f'<image>\n How aesthetically pleasing is the output image generated with the prompt "{prompt}"?'})
        messages.append({'role': 'assistant', 'content': f'I would rate this at a {prediction}'})

        # if np.random.rand() > 0.5:
        #     messages.append({'role': 'user', 'content': f'<image>\n Does this image match the style presented in the prompt "{prompt}"?'})
        #     messages.append({'role': 'assistant', 'content': f'Yes. This image properly matches the style of {args.lora_style}'})
        # else:
        #     wrong_style = diff_styles[np.random.randint(len(diff_styles))]
        #     wrong_prompt = f'{prompt_dataset[sample_id]['caption1']} in the style of {wrong_style}'
        #     messages.append({'role': 'user', 'content': f'<image>\n Does this image match the style presented in the prompt "{prompt}"?'})
        #     messages.append({'role': 'assistant', 'content': f'No. Instead of {wrong_style}, this image has style {args.lora_style}'})

        writer.write({'images': output_imgs, 'messages': messages})

        # if not os.path.exists(args.save_path):
        #     os.mkdir(args.save_path)
        # ind = len(os.listdir(args.save_path))
        # result.save(os.path.join(args.save_path, f"result_{ind}.png"))
        args.seed = args.seed + 1


if __name__ == "__main__":
    args = create_argparser().parse_args()
    device = get_device()
    dist.initialize_dist(device, args.dist_timeout)
    writer = MDSWriter(out = f'{remote}/aesthetic-rank{dist.get_global_rank()}', compression = "zstd", columns = columns)
    main(args)
    writer.finish()
