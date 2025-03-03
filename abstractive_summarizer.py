from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AbstractiveSummarizer:
    def __init__(self):
        """Initialize the abstractive summarizer with Pegasus model"""
        try:
            # Initialize NLTK
            nltk.download('punkt', quiet=True)
            
            # Load Pegasus model and tokenizer
            logger.info("Loading Pegasus model and tokenizer...")
            self.model_name = "Fryb14/pegasus-finetuned-summarization"
            self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
            self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name)
            
            # Set device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            
            # Initialize evaluation components
            self.sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
            self.smoothing = SmoothingFunction().method1
            
            logger.info("Initialization complete!")
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            raise

    def summarize(self, text, max_length=150, min_length=30, num_beams=4):
        """
        Generate abstractive summary using Pegasus model.
        
        Args:
            text (str): Input text to summarize
            max_length (int): Maximum length of the summary
            min_length (int): Minimum length of the summary
            num_beams (int): Number of beams for beam search
            
        Returns:
            str: Generated summary
        """
        try:
            if not text.strip():
                return ""

            # Tokenize input text
            inputs = self.tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
            inputs = inputs.to(self.device)

            # Generate summary
            start_time = time.time()
            summary_ids = self.model.generate(
                inputs["input_ids"],
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            
            # Decode summary
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            generation_time = time.time() - start_time
            logger.info(f"Summary generated in {generation_time:.2f} seconds")
            
            return summary

        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            return ""

    def evaluate_summary(self, original_text, summary):
        """
        Evaluate the generated summary using multiple metrics.
        
        Args:
            original_text (str): Original input text
            summary (str): Generated summary
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        try:
            metrics = {}
            
            # Calculate BLEU score
            ref_tokens = nltk.word_tokenize(original_text.lower())
            sum_tokens = nltk.word_tokenize(summary.lower())
            metrics['bleu'] = sentence_bleu([ref_tokens], sum_tokens, smoothing_function=self.smoothing)
            
            # Calculate semantic similarity using Sentence-BERT
            orig_embedding = self.sentence_transformer.encode(original_text, convert_to_tensor=True)
            sum_embedding = self.sentence_transformer.encode(summary, convert_to_tensor=True)
            similarity = cosine_similarity([orig_embedding.cpu().numpy()], [sum_embedding.cpu().numpy()])[0][0]
            metrics['semantic_similarity'] = similarity
            
            # Calculate BERTScore
            P, R, F1 = self.bert_scorer.score([summary], [original_text])
            metrics['bert_score'] = float(F1[0])
            
            # Calculate compression ratio
            metrics['compression_ratio'] = len(summary.split()) / len(original_text.split())
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {
                'bleu': 0,
                'semantic_similarity': 0,
                'bert_score': 0,
                'compression_ratio': 0
            }

def main():
    # Initialize summarizer
    print("Initializing summarizer...")
    summarizer = AbstractiveSummarizer()
    
    # Get input from user
    print("\nEnter the text you want to summarize (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    
    text = "\n".join(lines)
    
    if not text.strip():
        print("No input text provided. Exiting...")
        return
    
    # Get summary parameters from user
    try:
        max_length = int(input("\nEnter maximum summary length (default 150): ") or 150)
        min_length = int(input("Enter minimum summary length (default 30): ") or 30)
        num_beams = int(input("Enter number of beams for generation (default 4): ") or 4)
    except ValueError:
        print("Invalid input. Using default values.")
        max_length, min_length, num_beams = 150, 30, 4
    
    # Generate summary
    print("\nGenerating summary...")
    summary = summarizer.summarize(text, max_length=max_length, min_length=min_length, num_beams=num_beams)
    
    print("\nOriginal Text:")
    print(text)
    print("\nGenerated Summary:")
    print(summary)
    
    # Evaluate summary
    print("\nEvaluating summary...")
    metrics = summarizer.evaluate_summary(text, summary)
    print("\nEvaluation Metrics:")
    print(f"BLEU Score: {metrics['bleu']:.4f}")
    print(f"Semantic Similarity: {metrics['semantic_similarity']:.4f}")
    print(f"BERTScore: {metrics['bert_score']:.4f}")
    print(f"Compression Ratio: {metrics['compression_ratio']:.4f}")

if __name__ == "__main__":
    main()
