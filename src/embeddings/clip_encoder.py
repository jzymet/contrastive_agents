import torch
import numpy as np
from typing import List, Optional
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from PIL import Image
import requests
from io import BytesIO


class CLIPEncoder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.embedding_dim = 512
    
    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        all_embeddings = []
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding texts with CLIP")
        
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            inputs = self.processor(
                text=batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.device)
            
            text_embeds = self.model.get_text_features(**inputs)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            all_embeddings.append(text_embeds.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    @torch.no_grad()
    def encode_images(self, images: List, batch_size: int = 16, show_progress: bool = True) -> np.ndarray:
        all_embeddings = []
        
        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding images with CLIP")
        
        for i in iterator:
            batch_images = images[i:i + batch_size]
            inputs = self.processor(
                images=batch_images,
                return_tensors="pt"
            ).to(self.device)
            
            image_embeds = self.model.get_image_features(**inputs)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            all_embeddings.append(image_embeds.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def load_image_from_url(self, url: str) -> Optional[Image.Image]:
        try:
            response = requests.get(url, timeout=10)
            return Image.open(BytesIO(response.content)).convert("RGB")
        except Exception:
            return None
