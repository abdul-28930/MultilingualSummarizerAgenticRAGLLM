from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from summarizer import Summarizer
import spacy
import pytextrank  # For textrank
import yake
from dotenv import load_dotenv
import openai
import os

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key securely
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load Pegasus models
model_name = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_name)

# Load BERT Summarizer
bertsum_model = Summarizer()

# Load SpaCy and add pytextrank
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")  # Add TextRank to the SpaCy pipeline

# Keyword extraction using YAKE!
kw_extractor = yake.KeywordExtractor()

def agentic_rag_summarization(text):
    """
    Performs Agentic RAG summarization using Pegasus, BERTSUM, and TextRank.

    Args:
        text: The input text to be summarized.

    Returns:
        A dictionary containing the extracted entities, keywords,
        and summaries from each method.
    """
    # SpaCy processing
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]  # Named Entity Recognition (NER)
    keywords = [kw[0] for kw in kw_extractor.extract_keywords(text)]  # Keywords from YAKE!

    # Abstractive Summarization (PEGASUS)
    input_ids = tokenizer(
        text, truncation=True, padding="longest", return_tensors="pt"
    ).input_ids
    summary_ids = pegasus_model.generate(input_ids, max_length=60, min_length=10, length_penalty=2.0)
    pegasus_summary = tokenizer.decode(
        summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    # Extractive Summarization (BERTSUM)
    bertsum_summary = " ".join(bertsum_model(text, ratio=0.2))

    # Extractive Summarization (TextRank)
    textrank_summary = " ".join(
        [str(sent) for sent in doc._.textrank.summary(limit_sentences=3)]
    )

    return {
        "entities": entities,
        "keywords": keywords,
        "pegasus_summary": pegasus_summary,
        "bertsum_summary": bertsum_summary,
        "textrank_summary": textrank_summary,
    }


# Example Usage
if __name__ == "__main__":
    text = "Artificial Intelligence is transforming industries through natural language processing, robotics, and machine learning."
    results = agentic_rag_summarization(text)
    print("Entities:", results["entities"])
    print("Keywords:", results["keywords"])
    print("Pegasus Summary:", results["pegasus_summary"])
    print("BERTSUM Summary:", results["bertsum_summary"])
    print("TextRank Summary:", results["textrank_summary"])
