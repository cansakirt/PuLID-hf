"""PuLID is a tuning-free ID customization approach.
PuLID maintains high ID fidelity while effectively reducing interference
with the original model's behavior.

Code: https://github.com/ToTheBeginning/PuLID
Technical report: https://arxiv.org/abs/2404.16022

❗️❗️❗️ Tips:
- a single ID image is usually sufficient, you can also supplement with additional auxiliary images
- We offer two modes: fidelity mode and extremely style mode.
In most cases, the default fidelity mode should suffice.
If you find that the generated results are not stylized enough, you can choose the extremely style mode.
"""

import json
import gradio as gr
import numpy as np
import pandas as pd
import torch
import uuid

from pulid import attention_processor as attention
from pulid.pipeline import PuLIDPipeline
from pulid.utils import resize_numpy_image_long, seed_everything
from PIL import Image
import io

torch.set_grad_enabled(False)

pipeline = PuLIDPipeline()


# def test(
#     face_image,
#     supp_image1,
#     supp_image2,
#     supp_image3,
#     prompts,
#     neg_prompt,
#     scale,
#     n_samples,
#     seed,
#     steps,
#     H,
#     W,
#     id_scale,
#     mode,
#     id_mix,
# ):
#     face_image, supp_image1, supp_image2, supp_image3, prompts, neg_prompt, scale, n_samples, seed, steps, H, W, id_scale, mode, id_mix = (
#         inputs_with_prompts
#     )

#     supp_images = [supp_image1, supp_image2, supp_image3]
#     prompts = prompts.value.split("\n")  # Splitting text input into separate prompts
#     results = []
#     debug_img_lists = []
#     for prompt in prompts:
#         if prompt.strip():  # Check if the prompt is not just empty space
#             print(face_image, supp_images, prompt, neg_prompt, scale, n_samples, seed, steps, H, W, id_scale, mode, id_mix)
#             print("---------------")
#             results.append(prompt)
#             debug_img_lists.extend(prompt)
#     return results, debug_img_lists


DEFAULT_NEGATIVE_PROMPT = (
    "flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, worst quality,"
    "artifacts noise, text, watermark, glitch, deformed, mutated, ugly, disfigured, hands, "
    "low resolution, partially rendered objects, deformed or partially rendered eyes, "
    "deformed, deformed eyeballs, cross-eyed, blurry"
)

DEFAULT_PROMPTS = """portrait, flat papercut style, silhouette, clean cuts, paper, sharp edges, minimalist,color block,man
portrait, impressionist painting, loose brushwork, vibrant color, light and shadow play
portrait, superman
portrait, the legend of zelda, anime
portrait,cinematic,wolf ears,white hair
woman,cartoon,solo,Popmart Blind Box, Super Mario, 3d"""


def run(id_image, supp_images, prompt, neg_prompt, scale, n_samples, seed, steps, H, W, id_scale, mode, id_mix):
    pipeline.debug_img_list = []
    if mode == "fidelity":
        attention.NUM_ZERO = 8
        attention.ORTHO = False
        attention.ORTHO_v2 = True
    elif mode == "extremely style":
        attention.NUM_ZERO = 16
        attention.ORTHO = True
        attention.ORTHO_v2 = False
    else:
        raise ValueError("Unsupported mode")

    if id_image is not None:
        id_image = resize_numpy_image_long(id_image, 1024)
        id_embeddings = pipeline.get_id_embedding(id_image)
    else:
        id_embeddings = None

    # Support image handling
    for i, supp_id_image in enumerate(supp_images):
        if supp_id_image is not None:
            supp_id_image = resize_numpy_image_long(supp_id_image, 1024)
            supp_id_embeddings = pipeline.get_id_embedding(supp_id_image)
            if id_embeddings is not None:
                if id_mix:
                    id_embeddings = torch.cat((id_embeddings, supp_id_embeddings), dim=1)
                else:
                    id_embeddings = torch.cat((id_embeddings, supp_id_embeddings[:, :5]), dim=1)

    seed_everything(seed)
    ims = []
    for _ in range(n_samples):
        img = pipeline.inference(prompt, (1, H, W), neg_prompt, id_embeddings, id_scale, scale, steps)[0]
        ims.append(np.array(img))

    return ims, pipeline.debug_img_list


def run_batch(inputs_with_prompts):
    face_image, supp_image1, supp_image2, supp_image3, prompts, neg_prompt, scale, n_samples, seed, steps, H, W, id_scale, mode, id_mix = (
        inputs_with_prompts
    )

    supp_images = [supp_image1, supp_image2, supp_image3]
    prompts = prompts.value.split("\n")  # Splitting text input into separate prompts
    results = []
    debug_img_lists = []
    df_results = []
    for prompt in prompts:
        if prompt.strip():  # Ensure the prompt is not just empty space
            ims, debug_imgs = run(face_image, supp_images, prompt, neg_prompt, scale, n_samples, seed, steps, H, W, id_scale, mode, id_mix)
            row = {
                json.dumps(
                    {
                        "prompt": prompt,
                        "neg_prompt": neg_prompt,
                        "scale": scale,
                        "n_samples": n_samples,
                        "seed": seed,
                        "steps": steps,
                        "H": H,
                        "W": W,
                        "id_scale": id_scale,
                        "mode": mode,
                        "id_mix": id_mix,
                    }
                )
            }
            for i, img in enumerate(ims):
                img_id = str(uuid.uuid4())
                img_thumbnail = Image.fromarray(img).resize((100, 100))
                buffer = io.BytesIO()
                img_thumbnail.save(buffer, format="JPEG")
                row[f"Image {i+1} UUID"] = img_id
                row[f"Image {i+1} Render"] = buffer.getvalue()
            df_results.append(row)
            results.append(ims)
            debug_img_lists.extend(debug_imgs)
    df = pd.DataFrame(df_results)
    return results, debug_img_lists, df


with gr.Blocks(title="PuLID Batch Processor", css=".gr-box {border-color: #8136e2}") as demo:
    with gr.Row():
        with gr.Column():
            face_image = gr.Image(label="ID image (main)", sources="upload", type="numpy", height=256)
            with gr.Accordion("Auxiliary ID images", open=False):  # noqa: SIM117
                with gr.Row():
                    supp_image1 = gr.Image(label="Additional ID image (auxiliary)", sources="upload", type="numpy", height=256)
                    supp_image2 = gr.Image(label="Additional ID image (auxiliary)", sources="upload", type="numpy", height=256)
                    supp_image3 = gr.Image(label="Additional ID image (auxiliary)", sources="upload", type="numpy", height=256)
            prompts = gr.Textbox(
                label="Prompts (separated by new lines)",
                placeholder="Enter one prompt per line",
                lines=10,
                max_lines=20,
                value=DEFAULT_PROMPTS,
            )
            submit = gr.Button("Generate")
            neg_prompt = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEGATIVE_PROMPT)
            scale = gr.Slider(label="CFG, recommend value range [1, 1.5], 1 will be faster", value=1.2, minimum=1, maximum=1.5, step=0.1)
            n_samples = gr.Slider(label="Num samples", value=4, minimum=1, maximum=4, step=1)
            seed = gr.Slider(label="Seed", value=42, minimum=np.iinfo(np.uint32).min, maximum=np.iinfo(np.uint32).max, step=1)
            steps = gr.Slider(label="Steps", value=4, minimum=1, maximum=8, step=1)
            with gr.Row():
                H = gr.Slider(label="Height", value=1024, minimum=512, maximum=1280, step=64)
                W = gr.Slider(label="Width", value=768, minimum=512, maximum=1280, step=64)
            with gr.Row():
                id_scale = gr.Slider(label="ID scale", minimum=0, maximum=5, step=0.05, value=0.8, interactive=True)
                mode = gr.Dropdown(label="mode", choices=["fidelity", "extremely style"], value="fidelity")
                id_mix = gr.Checkbox(
                    label="ID Mix (if you want to mix two ID image, please turn this on, otherwise, turn this off)",
                    value=False,
                )

        with gr.Column():
            output_gallery = gr.Gallery(label="Generated Images", elem_id="gallery")
            debug_output_gallery = gr.Gallery(label="Debug Images", elem_id="gallery", visible=True)
            output_dataframe = gr.Dataframe(label="Output Parameters and Images")

    inputs_with_prompts = [
        face_image,
        supp_image1,
        supp_image2,
        supp_image3,
        prompts,
        neg_prompt,
        scale,
        n_samples,
        seed,
        steps,
        H,
        W,
        id_scale,
        mode,
        id_mix,
    ]
    # submit.click(fn=test, inputs=inputs_with_prompts, outputs=[output_gallery, debug_output_gallery, output_dataframe])
    submit.click(fn=run_batch, inputs=inputs_with_prompts, outputs=[output_gallery, debug_output_gallery, output_dataframe])


demo.launch(share=False, debug=True)
