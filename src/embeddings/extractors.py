import torch
import numpy as np
from typing import List, Optional
from tqdm import tqdm

from .base import BaseEmbeddingExtractor


class BERTExtractor(BaseEmbeddingExtractor):
    """BERT-base-uncased embedding extractor (anisotropic)."""
    
    def __init__(self, target_dim: int = 768):
        super().__init__(target_dim)
        self.model_name = 'bert'
        self.embedding_dim = 768
        self._load_model()
    
    def _load_model(self):
        from transformers import AutoModel, AutoTokenizer
        
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
    
    def _extract_raw(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        
        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]
                embedding = last_hidden.mean(dim=1).squeeze(0).cpu().numpy()
                embeddings.append(embedding)
        
        return np.stack(embeddings)


class RoBERTaExtractor(BaseEmbeddingExtractor):
    """RoBERTa-base embedding extractor (anisotropic)."""
    
    def __init__(self, target_dim: int = 768):
        super().__init__(target_dim)
        self.model_name = 'roberta'
        self.embedding_dim = 768
        self._load_model()
    
    def _load_model(self):
        from transformers import AutoModel, AutoTokenizer
        
        self.model = AutoModel.from_pretrained('roberta-base')
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
    
    def _extract_raw(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        
        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]
                embedding = last_hidden.mean(dim=1).squeeze(0).cpu().numpy()
                embeddings.append(embedding)
        
        return np.stack(embeddings)


class SimCSEExtractor(BaseEmbeddingExtractor):
    """SimCSE embedding extractor (contrastive, uniform)."""
    
    def __init__(self, target_dim: int = 768):
        super().__init__(target_dim)
        self.model_name = 'simcse'
        self.embedding_dim = 768
        self._load_model()
    
    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('princeton-nlp/sup-simcse-bert-base-uncased')
    
    def _extract_raw(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False)


class JinaExtractor(BaseEmbeddingExtractor):
    """Jina-embeddings-v3 extractor (contrastive, SOTA)."""
    
    def __init__(self, target_dim: int = 768):
        super().__init__(target_dim)
        self.model_name = 'jina'
        self.embedding_dim = 1024
        self._load_model()
    
    def _load_model(self):
        from transformers import AutoModel, AutoTokenizer
        
        try:
            self.model = AutoModel.from_pretrained(
                'jinaai/jina-embeddings-v3',
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v3')
            
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self.model.eval()
            self.use_sentence_transformer = False
        except Exception as e:
            print(f"Falling back to sentence-transformers for Jina: {e}")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('jinaai/jina-embeddings-v2-base-en')
            self.use_sentence_transformer = True
            self.embedding_dim = 768
    
    def _extract_raw(self, texts: List[str]) -> np.ndarray:
        if self.use_sentence_transformer:
            return self.model.encode(texts, show_progress_bar=False)
        
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                if hasattr(outputs, 'last_hidden_state'):
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
                else:
                    embedding = outputs[0].mean(dim=1).squeeze(0).cpu().numpy()
                embeddings.append(embedding)
        
        return np.stack(embeddings)


class DummyExtractor(BaseEmbeddingExtractor):
    """
    Dummy extractor for testing without loading actual models.
    Simulates uniform (contrastive) or anisotropic embeddings.
    """
    
    def __init__(self, target_dim: int = 768, is_uniform: bool = True, model_name: str = 'dummy'):
        super().__init__(target_dim)
        self.model_name = model_name
        self.embedding_dim = target_dim
        self.is_uniform = is_uniform
        self._base_direction = None
    
    def _load_model(self):
        pass
    
    def _extract_raw(self, texts: List[str]) -> np.ndarray:
        n = len(texts)
        
        if self.is_uniform:
            embeddings = np.random.randn(n, self.embedding_dim).astype(np.float32)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        else:
            if self._base_direction is None:
                self._base_direction = np.random.randn(1, self.embedding_dim).astype(np.float32)
                self._base_direction = self._base_direction / np.linalg.norm(self._base_direction)
            
            noise = np.random.randn(n, self.embedding_dim).astype(np.float32) * 0.3
            embeddings = self._base_direction + noise
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings


def get_extractor(model_name: str, target_dim: int = 768, use_dummy: bool = False) -> BaseEmbeddingExtractor:
    """
    Factory function to get embedding extractor by name.
    
    Args:
        model_name: One of 'bert', 'roberta', 'llama3', 'simcse', 'jina', 'llm2vec'
        target_dim: Target embedding dimension (default 768)
        use_dummy: Use dummy extractor for testing
        
    Returns:
        Embedding extractor instance
    """
    if use_dummy:
        is_uniform = model_name in ['simcse', 'jina', 'llm2vec']
        return DummyExtractor(target_dim=target_dim, is_uniform=is_uniform, model_name=model_name)
    
    extractors = {
        'bert': BERTExtractor,
        'roberta': RoBERTaExtractor,
        'simcse': SimCSEExtractor,
        'jina': JinaExtractor,
    }
    
    if model_name not in extractors:
        print(f"Model {model_name} not available, using dummy extractor")
        is_uniform = model_name in ['simcse', 'jina', 'llm2vec']
        return DummyExtractor(target_dim=target_dim, is_uniform=is_uniform, model_name=model_name)
    
    try:
        return extractors[model_name](target_dim=target_dim)
    except Exception as e:
        print(f"Failed to load {model_name}: {e}. Using dummy extractor.")
        is_uniform = model_name in ['simcse', 'jina', 'llm2vec']
        return DummyExtractor(target_dim=target_dim, is_uniform=is_uniform, model_name=model_name)


ANISOTROPIC_MODELS = ['bert', 'roberta', 'llama3']
CONTRASTIVE_MODELS = ['simcse', 'jina', 'llm2vec']
ALL_MODELS = ANISOTROPIC_MODELS + CONTRASTIVE_MODELS
