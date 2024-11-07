from transformers import BertTokenizer, BertModel, BertForSequenceClassification, pipeline
import torch
import numpy as np
from game_automation.i18n.internationalization import i18n

class AdvancedLanguageProcessor:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.sentiment_model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=self.sentiment_model, tokenizer=self.tokenizer)
        self.generator = pipeline('text-generation', model='gpt2')

    async def analyze_text(self, text):
        # 对输入文本进行编码
        encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        # 获取BERT模型的输出
        with torch.no_grad():
            output = self.model(**encoded_input)
        
        # 获取[CLS]标记的输出作为整个序列的表示
        sequence_output = output.last_hidden_state[:, 0, :].numpy()
        
        # 进行情感分析
        sentiment = self.sentiment_pipeline(text)[0]
        
        # 提取关键词
        keywords = self.extract_keywords(text)
        
        return {
            'embedding': sequence_output,
            'sentiment': sentiment,
            'keywords': keywords
        }

    def extract_keywords(self, text):
        # 使用TF-IDF或TextRank等算法提取关键词
        # 这里使用一个简单的实现，实际应用中可以使用更复杂的算法
        words = text.lower().split()
        word_freq = {}
        for word in words:
            if word not in self.tokenizer.vocab:
                continue
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
        return sorted(word_freq, key=word_freq.get, reverse=True)[:5]

    async def generate_response(self, prompt, max_length=50):
        generated_text = self.generator(prompt, max_length=max_length, num_return_sequences=1)[0]['generated_text']
        return generated_text

    async def translate(self, text, target_language):
        # 使用i18n模块进行翻译
        translated_text = i18n.translate(text, target_language)
        return translated_text

    async def analyze_dialogue(self, dialogue_history):
        # 分析整个对话历史
        full_text = " ".join(dialogue_history)
        analysis = await self.analyze_text(full_text)
        
        # 分析对话的连贯性
        coherence = self.analyze_coherence(dialogue_history)
        
        # 识别对话中的意图
        intent = self.identify_intent(dialogue_history[-1])
        
        return {
            'overall_analysis': analysis,
            'coherence': coherence,
            'last_utterance_intent': intent
        }

    def analyze_coherence(self, dialogue_history):
        # 简单的对话连贯性分析
        if len(dialogue_history) < 2:
            return 1.0  # 如果对话历史太短，假设是连贯的
        
        coherence_scores = []
        for i in range(1, len(dialogue_history)):
            prev_encoded = self.tokenizer(dialogue_history[i-1], return_tensors='pt')
            curr_encoded = self.tokenizer(dialogue_history[i], return_tensors='pt')
            
            with torch.no_grad():
                prev_output = self.model(**prev_encoded).last_hidden_state[:, 0, :]
                curr_output = self.model(**curr_encoded).last_hidden_state[:, 0, :]
            
            similarity = torch.cosine_similarity(prev_output, curr_output)
            coherence_scores.append(similarity.item())
        
        return np.mean(coherence_scores)

    def identify_intent(self, utterance):
        # 简单的意图识别，可以根据需要扩展
        intents = {
            'greeting': ['hello', 'hi', 'hey'],
            'farewell': ['goodbye', 'bye', 'see you'],
            'question': ['what', 'why', 'how', 'when', 'where'],
            'command': ['do', 'go', 'get', 'find']
        }
        
        utterance = utterance.lower()
        for intent, keywords in intents.items():
            if any(keyword in utterance for keyword in keywords):
                return intent
        
        return 'statement'  # 默认意图

advanced_language_processor = AdvancedLanguageProcessor()