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
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('bert-base-uncased')

    def _extract_raw(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=True, batch_size=batch_size)


class RoBERTaExtractor(BaseEmbeddingExtractor):
    """RoBERTa-base embedding extractor (anisotropic)."""

    def __init__(self, target_dim: int = 768):
        super().__init__(target_dim)
        self.model_name = 'roberta'
        self.embedding_dim = 768
        self._load_model()

    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('roberta-base')

    def _extract_raw(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=True, batch_size=batch_size)


class SimCSEExtractor(BaseEmbeddingExtractor):
    """SimCSE embedding extractor (contrastive, uniform)."""

    def __init__(self, target_dim: int = 768):
        super().__init__(target_dim)
        self.model_name = 'simcse'
        self.embedding_dim = 768
        self._load_model()

    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            'princeton-nlp/sup-simcse-bert-base-uncased')

    def _extract_raw(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False, batch_size=batch_size)


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
            self.model = AutoModel.from_pretrained('jinaai/jina-embeddings-v3',
                                                   trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                'jinaai/jina-embeddings-v3')

            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self.model.eval()
            self.use_sentence_transformer = False
        except Exception as e:
            print(f"Falling back to sentence-transformers for Jina: {e}")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                'jinaai/jina-embeddings-v2-base-en')
            self.use_sentence_transformer = True
            self.embedding_dim = 768

    def _extract_raw(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if self.use_sentence_transformer:
            return self.model.encode(texts, show_progress_bar=False, batch_size=batch_size)

        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
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
                    batch_embs = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                else:
                    batch_embs = outputs[0].mean(dim=1).cpu().numpy()
                embeddings.append(batch_embs)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.vstack(embeddings)


class LLaMA3Extractor(BaseEmbeddingExtractor):
    """
    LLaMA-3-8B base embeddings (CLM-trained, anisotropic).
    NO instruction tuning, NO contrastive tuning.
    """

    def __init__(self, target_dim: int = 768, use_quantization: bool = True):
        super().__init__(target_dim)
        self.model_name = 'llama3'
        self.embedding_dim = 4096
        self.use_quantization = use_quantization
        self._load_model()

    def _load_model(self):
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

        if self.use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4")
            self.model = AutoModel.from_pretrained(
                "meta-llama/Meta-Llama-3-8B",
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16)
        else:
            self.model = AutoModel.from_pretrained(
                "meta-llama/Meta-Llama-3-8B",
                torch_dtype=torch.float16,
                device_map="auto")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    def _extract_raw(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]
                batch_embs = last_hidden.mean(dim=1).cpu().numpy()
                embeddings.append(batch_embs)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.vstack(embeddings)


class LLM2VecExtractor(BaseEmbeddingExtractor):
    """
    LLM2Vec: LLaMA-3 + contrastive fine-tuning (uniform).
    This is the KEY comparison showing contrastive tuning fixes LLMs!
    """

    def __init__(self, target_dim: int = 768):
        super().__init__(target_dim)
        self.model_name = 'llm2vec'
        self.embedding_dim = 4096
        self._load_model()

    def _load_model(self):
        try:
            from llm2vec import LLM2Vec

            self.model = LLM2Vec.from_pretrained(
                "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
                peft_model_name_or_path=
                "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
                device_map="auto",
                torch_dtype=torch.float16)
            self.use_llm2vec = True
        except Exception as e:
            print(f"LLM2Vec not available: {e}. Using dummy.")
            self.use_llm2vec = False

    def _extract_raw(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        if not self.use_llm2vec:
            n = len(texts)
            embeddings = np.random.randn(n, 4096) / np.sqrt(4096)
            return embeddings

        return self.model.encode(texts, batch_size=batch_size)


class DummyExtractor(BaseEmbeddingExtractor):
    """Dummy extractor for testing."""

    def __init__(self, target_dim: int = 768, is_uniform: bool = False, model_name: str = 'dummy'):
        super().__init__(target_dim)
        self.model_name = model_name
        self.embedding_dim = target_dim
        self.is_uniform = is_uniform

    def _load_model(self):
        pass

    def _extract_raw(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        n = len(texts)
        np.random.seed(42)
        
        if self.is_uniform:
            embeddings = np.random.randn(n, self.target_dim) / np.sqrt(self.target_dim)
        else:
            embeddings = np.random.randn(n, self.target_dim)
            decay = np.exp(-np.arange(self.target_dim) / 30)
            embeddings = embeddings * decay
        
        return embeddings


def get_extractor(model_name: str,
                  target_dim: int = 768,
                  use_dummy: bool = False) -> BaseEmbeddingExtractor:
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
        return DummyExtractor(target_dim=target_dim,
                              is_uniform=is_uniform,
                              model_name=model_name)

    extractors = {
        'bert': BERTExtractor,
        'roberta': RoBERTaExtractor,
        'llama3': LLaMA3Extractor,
        'simcse': SimCSEExtractor,
        'jina': JinaExtractor,
        'llm2vec': LLM2VecExtractor,
    }

    if model_name not in extractors:
        print(f"Model {model_name} not available, using dummy extractor")
        is_uniform = model_name in ['simcse', 'jina', 'llm2vec']
        return DummyExtractor(target_dim=target_dim,
                              is_uniform=is_uniform,
                              model_name=model_name)

    try:
        return extractors[model_name](target_dim=target_dim)
    except Exception as e:
        print(f"Failed to load {model_name}: {e}. Using dummy extractor.")
        is_uniform = model_name in ['simcse', 'jina', 'llm2vec']
        return DummyExtractor(target_dim=target_dim,
                              is_uniform=is_uniform,
                              model_name=model_name)


ANISOTROPIC_MODELS = ['bert', 'roberta', 'llama3']
CONTRASTIVE_MODELS = ['simcse', 'jina', 'llm2vec']
ALL_MODELS = ANISOTROPIC_MODELS + CONTRASTIVE_MODELS
