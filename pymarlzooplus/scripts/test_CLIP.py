import argparse
from transformers import CLIPProcessor, CLIPModel
import cv2


def get_CLIP_preds(args):

    # Define the model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model = model.eval()

    # Define the text and image transformations
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Read the image
    image = cv2.imread(args.input_path)
    # Transform from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(args.prompts)

    # Preprocess the image and prompts
    inputs = processor(text=args.prompts, images=image, return_tensors="pt", padding=True)

    # Get predictions
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)

    print(probs)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--prompts", nargs="+", required=True,
                        help="The prompts to examined in the form 'prompt1', 'prompt2',... WITH spaces between prompts")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args_ = parse_args()
    print(args_)
    get_CLIP_preds(args_)

