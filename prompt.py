# make the response more structured
templatev0_1_chat = [
    {
        "role": "system",
        "content": """You are an intelligent bounding box generator for videos. You don't need to generate the videos themselves but need to generate the bounding boxes. I will provide you with a caption for a video with six frames, with two frames per second. Your task is to generate a list of realistic bounding boxes for the objects mentioned in the caption for each frame as well as a background keyword. The video frames are of size 512x512. The top-left corner has coordinates [0, 0]. The bottom-right corner has coordinnates [512, 512]. The bounding boxes should not overlap or go beyond the frame boundaries.

Each frame should be represented as `[{'id': unique object identifier incrementing from 0, 'name': object name, 'box': [box top-left x-coordinate, box top-left y-coordinate, box width, box height]}, ...]`. Each box should not include more than one object. Your generated frames must encapsulate the whole scenario depicted by the caption. Assume objects move and interact based on real-world physics, considering aspects such as gravity and elasticity. Assume the camera follows perspective geometry. Boxes for an object should have the same id across the frames, even if the object may disappear and reappear. If needed, you can make reasonable guesses. Provide a concise reasoning statement that is not longer than a few sentences before each generation. Refer to the examples below for the desired format. Never use markdown or other formats not in the examples. Do not start each frame with `-`. Do not include any comments in your response.""",
    },
    {
        "role": "user",
        "content": "Caption: A woman walking from the left to the right and a man jumping on the right in a room",
    },
    {
        "role": "assistant",
        "content": """Reasoning: A woman is walking from the left to the right so her x-coordinate should increase with her y-coordinate fixed. A man is jumping on the right so his x-coordinate should be large, and his y-coordinate should first decrease (upward movement) and then increase (downward movement due to gravity).
Frame 1: [{'id': 0, 'name': 'walking woman', 'box': [0, 270, 120, 200]}, {'id': 1, 'name': 'jumping man', 'box': [380, 290, 120, 180]}]
Frame 2: [{'id': 0, 'name': 'walking woman', 'box': [50, 270, 120, 200]}, {'id': 1, 'name': 'jumping man', 'box': [380, 205, 120, 200]}]
Frame 3: [{'id': 0, 'name': 'walking woman', 'box': [100, 270, 120, 200]}, {'id': 1, 'name': 'jumping man', 'box': [380, 175, 120, 200]}]
Frame 4: [{'id': 0, 'name': 'walking woman', 'box': [150, 270, 120, 200]}, {'id': 1, 'name': 'jumping man', 'box': [380, 175, 120, 200]}]
Frame 5: [{'id': 0, 'name': 'walking woman', 'box': [200, 270, 120, 200]}, {'id': 1, 'name': 'jumping man', 'box': [380, 205, 120, 200]}]
Frame 6: [{'id': 0, 'name': 'walking woman', 'box': [250, 270, 120, 200]}, {'id': 1, 'name': 'jumping man', 'box': [380, 290, 120, 180]}]
Background keyword: room""",
    },
    {
        "role": "user",
        "content": "Caption: A red ball is thrown from the left to the right in a garden",
    },
    {
        "role": "assistant",
        "content": """Reasoning: A ball is thrown from the left to the right, so its x-coordinate should increase. Due to gravity, its y-coordinate should increase, and the speed should be faster in later frames until it hits the ground. Due to its elasticity, the ball bounces back when it hits the ground.
Frame 1: [{'id': 0, 'name': 'red ball', 'box': [0, 206, 50, 50]}]
Frame 2: [{'id': 0, 'name': 'red ball', 'box': [80, 246, 50, 50]}]
Frame 3: [{'id': 0, 'name': 'red ball', 'box': [160, 326, 50, 50]}]
Frame 4: [{'id': 0, 'name': 'red ball', 'box': [240, 446, 50, 50]}]
Frame 5: [{'id': 0, 'name': 'red ball', 'box': [320, 366, 50, 50]}]
Frame 6: [{'id': 0, 'name': 'red ball', 'box': [400, 446, 50, 50]}]
Background keyword: garden""",
    },
    {"role": "user", "content": "Caption: The camera is moving away from a painting"},
    {
        "role": "assistant",
        "content": """Reasoning: Due to perspective geometry, the painting will be smaller in later timesteps as the distance between the camera and the object is larger.
Frame 1: [{'id': 0, 'name': 'painting', 'box': [156, 181, 200, 150]}]
Frame 2: [{'id': 0, 'name': 'painting', 'box': [166, 189, 180, 135]}]
Frame 3: [{'id': 0, 'name': 'painting', 'box': [176, 196, 160, 120]}]
Frame 4: [{'id': 0, 'name': 'painting', 'box': [186, 204, 140, 105]}]
Frame 5: [{'id': 0, 'name': 'painting', 'box': [196, 211, 120, 90]}]
Frame 6: [{'id': 0, 'name': 'painting', 'box': [206, 219, 100, 75]}]
Background keyword: room""",
    },
]

templates = {
    "v0.1": templatev0_1_chat,
}

template_versions = list(templates.keys())


def get_num_parsed_layout_frames(template_version):
    return 6


# 6 frames
required_lines = [f"Frame {i+1}:" for i in range(6)] + ["Background keyword:"]
required_lines_ast = [True] * 6 + [False]

strip_before = required_lines[0]

stop = "\n\n"

prompts_demo = [
    "A boy walks from left to right in a park",
    "A boy runs from left to right in a dense forest",
    "A boy jumps up and down in a house",
    "A boy sitts on a chair in a room",
    "A boy walk through a street in a city",
    "A boy swims in a wild river",
    "A girl walks from left to right in a park",
    "A girl runs from left to right in a dense forest",
    "A girl jumps up and down in a house",
    "A girl sits on a chair in a room",
    "A girl walks through a street in a city",
    "A girl swims in a wild river",
    "A cat walks from left to right in a park",
    "A cat runs from left to right in a dense forest",
    "A cat jumps up and down in a house",
    "A cat sits on a chair in a room",
    "A cat walks through a street in a city",
    "A cat swims in a wild river",
    "A dog walks from left to right in a park",
    "A dog runs from left to right in a dense forest",
    "A dog jumps up and down in a house",
    "A dog sits on a chair in a room",
    "A dog walks through a street in a city",
    "A dog swims in a wild river",
    "A sheep walks from left to right in a park",
    "A sheep runs from left to right in a dense forest",
    "A sheep jumps up and down in a house",
    "A sheep sits on a chair in a room",
    "A sheep walks through a street in a city",
    "A sheep swims in a wild river",
    "Loopy walks from left to right in a park",
    "Loopy runs from left to right in a dense forest",
    "Loopy jumps up and down in a house",
    "Loopy sits on a chair in a room",
    "Loopy walks through a street in a city",
    "Loopy swims in a wild river",
    "A man walks from left to right in a park",
    "A man runs from left to right in a dense forest",
    "A man jumps up and down in a house",
    "A man sits on a chair in a room",
    "A man walks through a street in a city",
    "A man swims in a wild river",
    "A woman walks from left to right in a park",
    "A woman runs from left to right in a dense forest",
    "A woman jumps up and down in a house",
    "A woman sits on a chair in a room",
    "A woman walks through a street in a city",
    "A woman swims in a wild river"
]

prompt_types = ["demo", "lvd"]

negative_prompt = (
    "dull, gray, unrealistic, colorless, blurry, low-quality, weird, abrupt"
)


def get_prompts(prompt_type, return_predicates=False):
    if prompt_type.startswith("lvd"):
        from utils.eval.lvd import get_lvd_full_prompts, get_lvd_full_prompt_predicates

        if return_predicates:
            prompts = get_lvd_full_prompt_predicates(prompt_type)
        else:
            prompts = get_lvd_full_prompts(prompt_type)
    elif prompt_type == "demo":
        assert (
            not return_predicates
        ), "Predicates are not supported for this prompt type"
        prompts = prompts_demo
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

    return prompts


if __name__ == "__main__":
    if True:
        prompt_type = "demo"

        assert prompt_type in prompt_types, f"prompt_type {prompt_type} does not exist"

        prompts = get_prompts(prompt_type)
        prompt = prompts[-1]
    else:
        prompt = input("Prompt: ")

    template_key = "v0.1"
    template = templates[template_key]

    if isinstance(template, list):
        template = (
            "\n\n".join([item["content"] for item in template])
            + "\n\nCaption: {prompt}\nReasoning:"
        )

    prompt_full = template.replace("{prompt}", prompt.strip().rstrip(".")).strip()
    print(prompt_full)

    if False:
        import json

        print(json.dumps(prompt_full.strip("\n")))
