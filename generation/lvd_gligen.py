from models.controllable_pipeline_text_to_video_synth import TextToVideoSDPipeline
# from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth import TextToVideoSDPipeline
from diffusers import DPMSolverMultistepScheduler
from models.unet_3d_condition import UNet3DConditionModel
from models.pipelines import encode
from utils import parse, vis
from utils.layout_make import latent_embed, png_image_process
from utils.latent_vis import single_vis, remove_background
from prompt import negative_prompt
import utils
import numpy as np
import torch
from PIL import Image
import os

version = "lvd-gligen"

# %%
# H, W are generation H and W. box_W and box_W are for scaling the boxes to [0, 1].
pipe, H, W, box_H, box_W = None, None, None, None, None


def init(base_model):
    global pipe, H, W, box_H, box_W
    if base_model == "modelscope256":
        model_key = "longlian/text-to-video-lvd-ms"
        H, W = 256, 256
        box_H, box_W = parse.size
    elif base_model == "zeroscope":
        model_key = "longlian/text-to-video-lvd-zs"
        H, W = 320, 576
        box_H, box_W = parse.size
    else:
        raise ValueError(f"Unknown base model: {base_model}")

    # pipe = TextToVideoSDPipeline.from_pretrained(
    #     model_key, trust_remote_code=True, torch_dtype=torch.float16
    # )
    pipe = TextToVideoSDPipeline.from_pretrained(
        model_key, trust_remote_code=False, torch_dtype=torch.float16
    )
    # The default one is DDIMScheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.enable_vae_slicing()

    # No auxiliary guidance
    pipe.guidance_models = None

    return H, W


# %%
upsample_scale, upsample_mode = 1, "bilinear"

# %%

# Seems like `enable_model_cpu_offload` performs deepcopy so `save_attn_to_dict` does not save the attn
cross_attention_kwargs = {
    # This is for visualizations
    # 'offload_cross_attn_to_cpu': True
}


# %%
def run(
    parsed_layout,
    seed,
    num_inference_steps=40,
    num_frames=16,
    gligen_scheduled_sampling_beta=1.0,
    repeat_ind=None,
    save_annotated_videos=False,
    save_formats=["gif", "joblib"],
):
    condition = parse.parsed_layout_to_condition(
        parsed_layout,
        tokenizer=pipe.tokenizer,
        height=box_H,
        width=box_W,
        num_condition_frames=num_frames,
        verbose=True,
    )
    prompt, bboxes, phrases, object_positions, token_map = (
        condition.prompt,
        condition.boxes,
        condition.phrases,
        condition.object_positions,
        condition.token_map,
    )

    if repeat_ind is not None:
        save_suffix = repeat_ind

    else:
        save_suffix = f"seed{seed}"

    save_path = f"{parse.img_dir}/video_{save_suffix}.gif"
    if os.path.exists(save_path):
        print(f"Skipping {save_path}")
        return

    print("Generating")
    generator = torch.Generator(device="cuda").manual_seed(seed)

    lvd_gligen_boxes = []
    lvd_gligen_phrases = []
    for i in range(num_frames):
        lvd_gligen_boxes.append(
            [
                bboxes_item[i]
                for phrase, bboxes_item in zip(phrases, bboxes)
                if bboxes_item[i] != [0.0, 0.0, 0.0, 0.0]
            ]
        )
        lvd_gligen_phrases.append(
            [
                phrase
                for phrase, bboxes_item in zip(phrases, bboxes)
                if bboxes_item[i] != [0.0, 0.0, 0.0, 0.0]
            ]
        )

    # image = Image.open("images/rembg/boy.png")
    # array = np.array(image)
    # images = image_embed(array)  # -> [24, (320, 576, 3)]
    # # 转为int8
    # images = np.array(images, dtype=np.uint8)
    # image_latents = torch.zeros((24, 1, 4, 40, 72), dtype=torch.float16)
    # for i in range(24):
    #     image_latents[i] = encode(pipe, images[i], generator)
    # image_latents = image_latents.permute(1, 2, 0, 3, 4)
    # image_latents = image_latents * 0.5 + torch.randn_like(image_latents) * 0.5
    # print("image_latents.shape = ", image_latents.shape)

    image = Image.open("images/rembg/" + phrases[0] + ".png")
    image = image.resize((512, 512))
    array = np.array(image, dtype=np.uint8)
    image = png_image_process(array)
    image_latent = encode(pipe, image, generator)
    print("image_latent.shape = ", image_latent.shape)
    # print("----------")
    # print("bbox:", bboxes)
    # print("phrases:", phrases)
    # print("----------")
    # bbox: [[[0.0, 0.390625, 0.09765625, 0.5859375], [0.04245881453804348, 0.4118544072690217, 0.14011506453804348, 0.6071669072690218], [0.08491762907608696, 0.43308381453804345, 0.18257387907608696, 0.6283963145380435], [0.12737644361413045, 0.45431322180706524, 0.22503269361413045, 0.6496257218070652], [0.16983525815217393, 0.47554262907608696, 0.26749150815217393, 0.6708551290760869], [0.2038032863451087, 0.4967720363451087, 0.3014595363451087, 0.6920845363451087], [0.22503269361413042, 0.5180014436141305, 0.3226889436141304, 0.7133139436141305], [0.2462621008831522, 0.5392308508831521, 0.3439183508831522, 0.7345433508831521], [0.26749150815217393, 0.5604602581521739, 0.36514775815217393, 0.7557727581521739], [0.28872091542119566, 0.5816896654211956, 0.38637716542119566, 0.7770021654211956], [0.3099503226902174, 0.6029190726902174, 0.4076065726902174, 0.7982315726902174], [0.33117972995923917, 0.6241484799592392, 0.42883597995923917, 0.8194609799592392], [0.3524091372282609, 0.6453778872282608, 0.4500653872282609, 0.8406903872282608], [0.3736385444972826, 0.6666072944972826, 0.4712947944972826, 0.8619197944972826], [0.3948679517663044, 0.68359375, 0.4925242017663044, 0.87890625], [0.41609735903532613, 0.68359375, 0.5137536090353261, 0.87890625], [0.43732676630434786, 0.68359375, 0.5349830163043479, 0.87890625], [0.45855617357336953, 0.68359375, 0.5562124235733695, 0.87890625], [0.47978558084239126, 0.68359375, 0.5774418308423913, 0.87890625], [0.5010149881114131, 0.6708600118885869, 0.5986712381114131, 0.8661725118885869], [0.5222443953804348, 0.6496306046195652, 0.6199006453804348, 0.8449431046195652], [0.5434738026494565, 0.6284011973505435, 0.6411300526494565, 0.8237136973505435], [0.5647032099184783, 0.6071717900815217, 0.6623594599184783, 0.8024842900815217], [0.5859326171875, 0.5859423828125, 0.6835888671875, 0.7812548828125]]]
    # phrases: ['boy']
    torch.save(image_latent, f"tmp/" + phrases[0] + ".pt")

    # bear = torch.load("tmp/bear.pt", map_location="cpu")
    image_latent, _ = remove_background(image_latent)
    image_latent = image_latent.to("cuda")
    noise = latent_embed(image_latent, parsed_layout=bboxes[0], fps=24, generator=generator, gap=1)
    noise = noise.to(torch.float16)

    video_frames = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        height=H,
        width=W,
        num_frames=num_frames,
        cross_attention_kwargs=cross_attention_kwargs,
        generator=generator,
        lvd_gligen_scheduled_sampling_beta=gligen_scheduled_sampling_beta,
        lvd_gligen_boxes=lvd_gligen_boxes,
        lvd_gligen_phrases=lvd_gligen_phrases,
        latents=noise,
    ).frames
    # `diffusers` has a backward-breaking change
    # video_frames = (video_frames[0] * 255.).astype(np.uint8)

    # %%

    if save_annotated_videos:
        annotated_frames = [
            np.array(
                utils.draw_box(
                    Image.fromarray(video_frame), [bbox[i] for bbox in bboxes], phrases
                )
            )
            for i, video_frame in enumerate(video_frames)
        ]
        vis.save_frames(
            f"{save_path}/video_seed{seed}_with_box",
            frames=annotated_frames,
            formats="gif",
        )

    vis.save_frames(
        f"{parse.img_dir}/video_{save_suffix}", video_frames, formats=save_formats
    )