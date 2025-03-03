from sentence_transformers import SentenceTransformer
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from bert_score import BERTScorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExtractiveSummarizer:
    def __init__(self):
        # Initialize NLTK
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Initialize models
        self.sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        self.smoothing = SmoothingFunction().method1

    def summarize(self, text, num_sentences=3):
        """
        Performs extractive summarization using Sentence-BERT and TextRank.
        
        Args:
            text (str): The input text to be summarized
            num_sentences (int): Number of sentences in the summary
            
        Returns:
            str: The extractive summary
        """
        if not text.strip():
            return ""

        # Tokenize into sentences
        sentences = nltk.sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text

        # Generate sentence embeddings
        try:
            sentence_embeddings = self.sentence_transformer.encode(sentences)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return " ".join(sentences[:num_sentences])

        # Create similarity matrix
        similarity_matrix = cosine_similarity(sentence_embeddings)

        # Apply TextRank
        try:
            graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(graph)
            
            # Rank and select sentences
            ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
            summary = " ".join([s for _, s in ranked_sentences[:num_sentences]])
            return summary
        except Exception as e:
            logger.error(f"Error in TextRank algorithm: {e}")
            return " ".join(sentences[:num_sentences])

    def evaluate_summary(self, original_text, summary):
        """
        Evaluates the summary using multiple metrics.
        
        Args:
            original_text (str): The original text
            summary (str): The generated summary
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        metrics = {}
        
        try:
            # Calculate BLEU score
            ref_tokens = nltk.word_tokenize(original_text.lower())
            sum_tokens = nltk.word_tokenize(summary.lower())
            metrics['bleu'] = sentence_bleu([ref_tokens], sum_tokens, smoothing_function=self.smoothing)
            
            # Calculate semantic similarity
            orig_embedding = self.sentence_transformer.encode(original_text, convert_to_tensor=True)
            sum_embedding = self.sentence_transformer.encode(summary, convert_to_tensor=True)
            similarity = cosine_similarity([orig_embedding.cpu().numpy()], [sum_embedding.cpu().numpy()])[0][0]
            metrics['semantic_similarity'] = similarity
            
            # Calculate BERTScore
            P, R, F1 = self.bert_scorer.score([summary], [original_text])
            metrics['bert_score'] = float(F1[0])
            
            return metrics
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            return {'bleu': 0, 'semantic_similarity': 0, 'bert_score': 0}

def main():
    # Example usage
    text = """
    The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 km2 (2,700,000 sq mi), of which 5,500,000 km2 (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname and French Guiana. The Amazon represents over half of the planet's remaining rainforests, and comprises the largest and most biodiverse tract of tropical rainforest in the world, with an estimated 390 billion individual trees divided into 16,000 species.
    """
    
    # Initialize summarizer
    summarizer = ExtractiveSummarizer()
    
    # Generate summary
    print("Generating summary...")
    summary = summarizer.summarize(text, num_sentences=3)
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

if __name__ == "__main__":
    main()
