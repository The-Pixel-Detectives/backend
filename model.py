import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import CLIPVisionModelWithProjection, CLIPTokenizer, CLIPTextModelWithProjection, AutoModel
from PIL import Image


class Clip4Clip:
    def __init__(self):
        # Define image preprocessing
        self.preprocess = Compose([
            Resize(224),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        self.model = CLIPVisionModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k")
        self.model = self.model.eval()

        self.text_model = CLIPTextModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k")
        self.text_model = self.text_model.eval()
        self.tokenizer = CLIPTokenizer.from_pretrained("Searchium-ai/clip4clip-webvid150k")

    # Function to extract embeddings from an image
    def extract_embedding(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.preprocess(Image.fromarray(image).convert("RGB"))
        with torch.no_grad():
            visual_output = self.model(image.unsqueeze(0))

        # Normalizing the embeddings
        visual_output = visual_output["image_embeds"].squeeze()
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        return visual_output.numpy()

    def extract_text_embedding(self, text):
        inputs = self.tokenizer(text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            outputs = outputs["text_embeds"].squeeze()

        # Normalize embeddings for retrieval
        final_output = outputs / outputs.norm(dim=-1, keepdim=True)
        final_output = final_output.cpu().detach().numpy()
        return final_output


class JinaCLIP:
    def __init__(self):
        self.model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)

    # Function to extract embeddings from an image
    def extract_embedding(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        with torch.no_grad():
            visual_output = self.model.encode_image([image])

        # Normalizing the embeddings
        visual_output = visual_output.squeeze()
        visual_output = visual_output / np.linalg.norm(visual_output)
        return visual_output

    def extract_text_embedding(self, text):
        with torch.no_grad():
            outputs = self.model.encode_text([text])
            outputs = outputs.squeeze()

        # Normalize embeddings for retrieval
        outputs = outputs / np.linalg.norm(outputs)
        return outputs
