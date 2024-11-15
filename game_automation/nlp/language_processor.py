from utils.logger import detailed_logger
from utils.config_manager import config_manager

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

class LanguageProcessor:
    def __init__(self):
        if not HAS_SPACY:
            raise ImportError(
                "Spacy is required for NLP functionality. "
                "Please install NLP dependencies with: "
                "pip install -r requirements-nlp.txt"
            )
            
        self.logger = detailed_logger
        self.config = config_manager.get('nlp', {})
        self.nlp = spacy.load("en_core_web_sm")

    async def analyze_text(self, text):
        """分析文本并提取关键信息"""
        doc = self.nlp(text)
        
        analysis = {
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'noun_phrases': [chunk.text for chunk in doc.noun_chunks],
            'verbs': [token.lemma_ for token in doc if token.pos_ == "VERB"],
            'sentences': [sent.text for sent in doc.sents],
            'dependencies': [(token.text, token.dep_, token.head.text) for token in doc],
            'pos_tags': [(token.text, token.pos_) for token in doc]
        }
        
        self.logger.debug(f"Text analysis completed: {analysis}")
        return analysis

    async def extract_commands(self, text):
        """提取文本中的命令和动作"""
        doc = self.nlp(text)
        commands = []
        
        for sent in doc.sents:
            root = [token for token in sent if token.dep_ == "ROOT"][0]
            if root.pos_ == "VERB":
                obj = [token for token in root.children if token.dep_ in ["dobj", "pobj"]]
                if obj:
                    commands.append(f"{root.lemma_}_{obj[0].lemma_}")
                else:
                    commands.append(root.lemma_)
        
        return commands

    async def identify_context(self, text):
        """识别文本的上下文信息"""
        doc = self.nlp(text)
        context = {
            'location': [],
            'time': [],
            'objects': [],
            'actions': []
        }
        
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                context['location'].append(ent.text)
            elif ent.label_ in ["TIME", "DATE"]:
                context['time'].append(ent.text)
        
        for token in doc:
            if token.pos_ == "NOUN":
                context['objects'].append(token.text)
            elif token.pos_ == "VERB":
                context['actions'].append(token.lemma_)
        
        return context

    async def generate_response(self, text):
        """生成对文本的响应"""
        doc = self.nlp(text)
        
        # 简单的响应生成逻辑
        if any(token.text.lower() in ["hello", "hi"] for token in doc):
            return "Hello! How can I help you?"
        
        if "?" in text:
            return "I'll look into that for you."
        
        return "I understand your request."

    async def classify_intent(self, text):
        """分类文本的意图"""
        doc = self.nlp(text)
        
        intents = {
            'command': 0,
            'question': 0,
            'statement': 0,
            'request': 0
        }
        
        # 基于语法特征的简单意图分类
        if text.endswith("?"):
            intents['question'] = 1
        elif any(token.dep_ == "ROOT" and token.pos_ == "VERB" for token in doc):
            intents['command'] = 1
        elif any(token.text.lower() in ["please", "could", "would"] for token in doc):
            intents['request'] = 1
        else:
            intents['statement'] = 1
        
        return max(intents.items(), key=lambda x: x[1])[0]

language_processor = LanguageProcessor()
