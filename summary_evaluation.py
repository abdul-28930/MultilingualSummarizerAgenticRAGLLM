from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
import nltk
import numpy as np
import torch
from bert_score import BERTScorer
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set NLTK data path to a local directory
nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download required NLTK data with retry mechanism
def download_nltk_data():
    max_retries = 3
    for attempt in range(max_retries):
        try:
            nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                logger.warning(f"Failed to download NLTK data after {max_retries} attempts: {e}")
                return False
            continue

# Try to download NLTK data
download_nltk_data()

class SummaryEvaluator:
    def __init__(self):
        try:
            self.sentence_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {e}")
            self.sentence_transformer = None
        
        try:
            self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        except Exception as e:
            logger.warning(f"Failed to load BERTScorer: {e}")
            self.bert_scorer = None
        
        self.smoothing = SmoothingFunction().method1

    def tokenize_safely(self, text):
        """Safely tokenize text with fallback to basic splitting."""
        try:
            return nltk.word_tokenize(text.lower())
        except Exception as e:
            logger.warning(f"NLTK tokenization failed: {e}")
            return text.lower().split()

    def calculate_bleu_score(self, reference, summary):
        """Calculate BLEU score."""
        try:
            reference_tokens = self.tokenize_safely(reference)
            summary_tokens = self.tokenize_safely(summary)
            return round(sentence_bleu([reference_tokens], summary_tokens, smoothing_function=self.smoothing), 4)
        except Exception as e:
            logger.error(f"Error calculating BLEU score: {e}")
            return 0

    def calculate_semantic_similarity(self, reference, summary):
        """Calculate semantic similarity using Sentence Transformers."""
        if not self.sentence_transformer:
            return 0
            
        try:
            # Encode sentences
            reference_embedding = self.sentence_transformer.encode(reference, convert_to_tensor=True)
            summary_embedding = self.sentence_transformer.encode(summary, convert_to_tensor=True)
            
            # Calculate cosine similarity
            cosine_similarity = torch.nn.functional.cosine_similarity(reference_embedding.unsqueeze(0), 
                                                                    summary_embedding.unsqueeze(0))
            return round(float(cosine_similarity), 4)
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0

    def calculate_bert_score(self, reference, summary):
        """Calculate BERTScore."""
        if not self.bert_scorer:
            return 0
            
        try:
            P, R, F1 = self.bert_scorer.score([summary], [reference])
            return round(float(F1[0]), 4)
        except Exception as e:
            logger.error(f"Error calculating BERTScore: {e}")
            return 0

    def evaluate_summary(self, reference, summary):
        """Evaluate summary using multiple metrics."""
        if not reference or not summary:
            return {
                'bleu': 0,
                'semantic_similarity': 0,
                'bert_score': 0
            }

        metrics = {}
        
        # BLEU score
        bleu = self.calculate_bleu_score(reference, summary)
        metrics['bleu'] = bleu
        
        # Semantic similarity
        semantic_sim = self.calculate_semantic_similarity(reference, summary)
        metrics['semantic_similarity'] = semantic_sim
        
        # BERTScore
        bert_score = self.calculate_bert_score(reference, summary)
        metrics['bert_score'] = bert_score
        
        return metrics

    def get_overall_score(self, metrics):
        """Calculate an overall score from all metrics."""
        if not metrics:
            return 0.0
            
        weights = {
            'bleu': 0.3,
            'semantic_similarity': 0.35,
            'bert_score': 0.35
        }
        
        total_score = 0
        total_weight = 0
        
        for metric, value in metrics.items():
            if metric in weights and value is not None:
                total_score += value * weights[metric]
                total_weight += weights[metric]
        
        if total_weight == 0:
            return 0.0
            
        return round(total_score / total_weight, 4)
