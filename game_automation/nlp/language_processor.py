import spacy
import nltk
from transformers import pipeline
from utils.logger import setup_logger
from utils.config_manager import config_manager

class LanguageProcessor:
    def __init__(self):
        self.logger = setup_logger('language_processor')
        self.config = config_manager.get('nlp', {})
        
        # 加载spaCy模型
        self.nlp = spacy.load("en_core_web_sm")
        
        # 初始化NLTK
        nltk.download('punkt')
        nltk.download('wordnet')
        
        # 加载Hugging Face的文本生成模型
        self.text_generator = pipeline("text-generation", model="gpt2")

    async def analyze_text(self, text):
        doc = self.nlp(text)
        analysis = {
            "entities": [(ent.text, ent.label_) for ent in doc.ents],
            "noun_phrases": [chunk.text for chunk in doc.noun_chunks],
            "verbs": [token.lemma_ for token in doc if token.pos_ == "VERB"]
        }
        self.logger.info(f"Text analysis results: {analysis}")
        return analysis

    async def generate_response(self, prompt, max_length=50):
        generated_text = self.text_generator(prompt, max_length=max_length, num_return_sequences=1)[0]['generated_text']
        self.logger.info(f"Generated response: {generated_text}")
        return generated_text

    async def extract_keywords(self, text):
        doc = self.nlp(text)
        keywords = [token.text for token in doc if not token.is_stop and token.is_alpha]
        self.logger.info(f"Extracted keywords: {keywords}")
        return keywords

    async def sentiment_analysis(self, text):
        doc = self.nlp(text)
        sentiment = doc.sentiment
        self.logger.info(f"Sentiment analysis result: {sentiment}")
        return sentiment

    async def summarize_text(self, text, sentences=3):
        doc = self.nlp(text)
        sentences = list(doc.sents)
        summary = " ".join([str(sent) for sent in sentences[:sentences]])
        self.logger.info(f"Text summary: {summary}")
        return summary

language_processor = LanguageProcessor()