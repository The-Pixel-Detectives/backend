import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import CLIPVisionModelWithProjection, CLIPTokenizer, CLIPTextModelWithProjection, AutoModel
from PIL import Image

from sbir.options import Option
from sbir.model.model import Model
from sbir.utils.util import load_checkpoint
from sbir.data_utils.utils import preprocess
import torch.nn.functional as F


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
        self.device = torch.device("mps")
        self.model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)
        self.model = self.model.to(self.device)

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


class SBIRModel:
    def __init__(self):
        args = Option().parse()
        model = Model(args)
        checkpoint = load_checkpoint(args.load)
        cur = model.state_dict()
        new = {k: v for k, v in checkpoint['model'].items() if k in cur.keys()}
        cur.update(new)
        model.load_state_dict(cur)
        model.eval()
        self.model = model.cpu()

    def extract_embedding(self, image_path):
        im = preprocess(image_path=image_path, img_type="im")
        im = im.unsqueeze(0)
        with torch.no_grad():
            im, im_idxs = self.model(im, None, 'test', only_sa=True)
        im = im.squeeze()
        im = im[:1]
        im = F.normalize(im)
        im = im[0]
        return im.cpu().detach().numpy()

    def extract_sketch_embedding(self, image_path):
        sk = preprocess(image_path=image_path, img_type="sk")
        sk = sk.unsqueeze(0)
        with torch.no_grad():
            sk, sk_idxs = self.model(sk, None, 'test', only_sa=True)
        sk = sk.squeeze()
        sk = sk[:1]
        sk = F.normalize(sk)
        sk = sk[0]
        return sk.cpu().detach().numpy()


class SigLIP_MCIP:
    def __init__(self):
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms("ViT-SO400M-14-SigLIP-384")

        checkpoint_path = '/Volumes/CSZoneT7/AIC/data/checkpoints/MCIP-ViT-SO400M-14-SigLIP-384.pth'
        mcip_state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(mcip_state_dict, strict=True)
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer('ViT-SO400M-14-SigLIP-384')

    # Function to extract embeddings from an image
    def extract_embedding(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        with torch.no_grad():
            visual_output = self.model.encode_image(self.preprocess(image).unsqueeze(0))

        # Normalizing the embeddings
        visual_output /= visual_output.norm(dim=-1, keepdim=True)
        visual_output = visual_output.squeeze()
        return visual_output.detach().cpu().numpy()

    def extract_text_embedding(self, text):
        text = self.tokenizer([text])

        with torch.no_grad():
            outputs = self.model.encode_text(text)
            outputs /= outputs.norm(dim=-1, keepdim=True)
            outputs = outputs.squeeze()

        return outputs.detach().cpu().numpy()




# jina_model = JinaCLIP()
embedding_model = SigLIP_MCIP()
