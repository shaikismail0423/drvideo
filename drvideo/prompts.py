"""Prompts, adapted from the paper (Sec. 3 and the supplementary material)."""

# --- Vision prompts (sent with a frame) -------------------------------------

CAPTION = "Describe the picture in no more than 50 words."

# Augmentation type A: detailed caption.
DETAILED_CAPTION = "Give a detailed description of this image. What is shown in this image?"

# Augmentation type B: visual question answering. Also the initial augmentation prompt.
VQA = (
    "If there are factual errors in the question, provide a precise description "
    "of the image; if not, proceed to answer the question: {question}"
)

# --- Agent prompts (text only) ----------------------------------------------

HEADER = """You are given some language descriptions of a video along with a question about the video.
1. The video is {duration} long, containing a total of {total} frames.
2. Each sentence in these language descriptions represents the text description for a single frame.
3. The format of each sentence is frame id, description. The frame id indicates the temporal position of the frame, ranging from 1 to {total}.
Here are the descriptions of this video:
{document}
"""

PLANNING = HEADER + """
Here is the question: {question}
Here is the memory: {memory}
Your task is to determine whether these descriptions above can answer the question accurately, reasonably, and without contradiction.
If your answer is yes, give a reasonable explanation. The output will be as follows:
{{"confidence": "1", "explanation": "xxxx"}}
If your answer is no, the confidence is 0, indicating the provided information is insufficient. Give a reasonable explanation for what frame is missing. For each frame identified as potentially relevant, provide a concise description focusing on essential visual elements (e.g., objects, humans, interactions, actions, and scenes) in the explanation. The output will be as follows:
{{"confidence": "0", "explanation": "xxxx"}}
You must not provide any other response or explanation."""

INTERACTION = HEADER + """
Here is the memory: {memory}
To answer the following question: {question}
These descriptions are insufficient and cannot answer this question accurately, reasonably, and without contradiction.
Your task is to determine which frame needs which type of information so that the question can be answered accurately, reasonably, and without contradiction.
The two types of information are as follows:
A: Given an image, get a detailed description of the image (image caption, just like what is shown in this image?)
B: Given an image, get a response to the above question (visual question answering)
Please note that frame selections range from 1 to {total}. Select at most {max_frames} frames.
These frames already have type A information: {have_a}. These frames already have type B information: {have_b}. Do not select the same type of information for these frames again.
The key "frame" must be a single number. The output must be as follows:
{{"frames": [{{"frame": 1, "type": "A"}}, {{"frame": 2, "type": "B"}}]}}"""

ANSWERING = HEADER + """
Please answer the following question: {question}
{choices}
Please think step by step. The confidence must be 1, 2 or 3 (3 is the most confident), and the explanation must be one concise sentence. The output must be in the following format. You must not provide any other response.
{{"reasoning": "step by step thinking", "final_answer": "xxx", "confidence": "xxx", "explanation": "xxx"}}"""

CHOICES = """Here are the choices:
{options}
Your final answer must be the letter of one of the choices."""
