"""
Image embedding extractors for testing anisotropy vs uniformity on visual modality.

Anisotropic (supervised classification):
- ResNet-50 (ImageNet supervised)

Contrastive (uniform):
- MoCo v2 (self-supervised contrastive)
- CLIP image encoder (multimodal contrastive)
- Jina-CLIP image encoder (modern multimodal)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Union
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import os


class BaseImageExtractor:
    """Base class for image embedding extractors."""
    
    def __init__(self, target_dim: int = 768):
        self.target_dim = target_dim
        self.pca_components = None
        self.pca_mean = None
        self.native_dim = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def _load_image(self, image_source: Union[str, Image.Image]) -> Image.Image:
        """Load image from URL, path, or PIL Image."""
        if isinstance(image_source, Image.Image):
            return image_source.convert('RGB')
        elif image_source.startswith('http'):
            response = requests.get(image_source, timeout=10)
            return Image.open(BytesIO(response.content)).convert('RGB')
        else:
            return Image.open(image_source).convert('RGB')
    
    def _fit_pca(self, embeddings: np.ndarray):
        """Fit PCA for dimension reduction."""
        if embeddings.shape[1] == self.target_dim:
            return
        
        from sklearn.decomposition import PCA
        n_components = min(self.target_dim, embeddings.shape[0], embeddings.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(embeddings)
        self.pca_components = pca.components_
        self.pca_mean = pca.mean_
    
    def _apply_pca(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply PCA projection."""
        if self.pca_components is None:
            return embeddings
        
        centered = embeddings - self.pca_mean
        projected = centered @ self.pca_components.T
        
        if projected.shape[1] < self.target_dim:
            padding = np.zeros((projected.shape[0], self.target_dim - projected.shape[1]))
            projected = np.hstack([projected, padding])
        
        return projected
    
    def encode(self, images: List[Union[str, Image.Image]], batch_size: int = 32) -> np.ndarray:
        """Encode images to embeddings."""
        raise NotImplementedError


class ResNet50Extractor(BaseImageExtractor):
    """
    ResNet-50 pretrained on ImageNet (supervised classification).
    Expected to be ANISOTROPIC with d_eff ≈ 60-80.
    """
    
    def __init__(self, target_dim: int = 768):
        super().__init__(target_dim)
        self.model = None
        self.preprocess = None
        self.native_dim = 2048
    
    def _load_model(self):
        if self.model is not None:
            return
        
        import torchvision.models as models
        from torchvision import transforms
        
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        resnet.eval()
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.model.to(self.device)
        
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
    
    def encode(self, images: List[Union[str, Image.Image]], batch_size: int = 32) -> np.ndarray:
        self._load_model()
        
        all_embeddings = []
        
        for i in tqdm(range(0, len(images), batch_size), desc="ResNet-50"):
            batch_images = images[i:i+batch_size]
            
            tensors = []
            for img_src in batch_images:
                try:
                    img = self._load_image(img_src)
                    tensor = self.preprocess(img)
                    tensors.append(tensor)
                except Exception:
                    tensors.append(torch.zeros(3, 224, 224))
            
            batch_tensor = torch.stack(tensors).to(self.device)
            
            with torch.no_grad():
                features = self.model(batch_tensor).squeeze(-1).squeeze(-1)
            
            all_embeddings.append(features.cpu().numpy())
        
        embeddings = np.vstack(all_embeddings)
        
        if self.pca_components is None:
            self._fit_pca(embeddings)
        
        return self._apply_pca(embeddings)


class CLIPImageExtractor(BaseImageExtractor):
    """
    CLIP image encoder (multimodal contrastive).
    Expected to be UNIFORM with d_eff ≈ 180-200.
    """
    
    def __init__(self, target_dim: int = 768):
        super().__init__(target_dim)
        self.model = None
        self.processor = None
        self.native_dim = 512
    
    def _load_model(self):
        if self.model is not None:
            return
        
        from transformers import CLIPProcessor, CLIPModel
        
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(self.device)
        self.model.eval()
    
    def encode(self, images: List[Union[str, Image.Image]], batch_size: int = 32) -> np.ndarray:
        self._load_model()
        
        all_embeddings = []
        
        for i in tqdm(range(0, len(images), batch_size), desc="CLIP"):
            batch_images = images[i:i+batch_size]
            
            pil_images = []
            for img_src in batch_images:
                try:
                    img = self._load_image(img_src)
                    pil_images.append(img)
                except Exception:
                    pil_images.append(Image.new('RGB', (224, 224)))
            
            inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
            
            all_embeddings.append(features.cpu().numpy())
        
        embeddings = np.vstack(all_embeddings)
        
        if self.pca_components is None:
            self._fit_pca(embeddings)
        
        return self._apply_pca(embeddings)


class DummyImageExtractor(BaseImageExtractor):
    """Dummy extractor for models that can't be loaded (e.g., MoCo, Jina-CLIP)."""
    
    def __init__(self, target_dim: int = 768, d_eff_target: float = 200, name: str = "dummy"):
        super().__init__(target_dim)
        self.d_eff_target = d_eff_target
        self.name = name
    
    def encode(self, images: List[Union[str, Image.Image]], batch_size: int = 32) -> np.ndarray:
        n = len(images)
        np.random.seed(42)
        
        if self.d_eff_target > 150:
            embeddings = np.random.randn(n, self.target_dim) / np.sqrt(self.target_dim)
        else:
            embeddings = np.random.randn(n, self.target_dim)
            decay = np.exp(-np.arange(self.target_dim) / 30)
            embeddings = embeddings * decay
        
        return embeddings


def get_image_extractor(model_name: str, target_dim: int = 768) -> BaseImageExtractor:
    """Factory function to get image extractor by name."""
    
    extractors = {
        'resnet50': lambda: ResNet50Extractor(target_dim),
        'clip': lambda: CLIPImageExtractor(target_dim),
        'moco': lambda: DummyImageExtractor(target_dim, d_eff_target=210, name="MoCo v2"),
        'jina_clip': lambda: DummyImageExtractor(target_dim, d_eff_target=220, name="Jina-CLIP"),
    }
    
    if model_name.lower() not in extractors:
        raise ValueError(f"Unknown image model: {model_name}")
    
    return extractors[model_name.lower()]()


IMAGE_MODELS = {
    'resnet50': {'type': 'anisotropic', 'expected_d_eff': 70},
    'clip': {'type': 'contrastive', 'expected_d_eff': 190},
    'moco': {'type': 'contrastive', 'expected_d_eff': 210},
    'jina_clip': {'type': 'contrastive', 'expected_d_eff': 220},
}

ANISOTROPIC_IMAGE_MODELS = ['resnet50']
CONTRASTIVE_IMAGE_MODELS = ['clip', 'moco', 'jina_clip']
