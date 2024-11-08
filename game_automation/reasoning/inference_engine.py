from pyDatalog import pyDatalog
from utils.logger import detailed_logger

class InferenceEngine:
    def __init__(self):
        self.logger = detailed_logger
        pyDatalog.create_terms('X, Y, Z, present_in, mentioned_in, related_to, implies')
        self.facts = set()

    async def add_fact(self, subject, relation, object):
        """添加一个新的事实到推理引擎"""
        fact = (subject, relation, object)
        self.facts.add(fact)
        # 使用pyDatalog添加事实
        pyDatalog.assert_fact(relation, subject, object)
        self.logger.debug(f"Added fact: {subject} {relation} {object}")

    async def query(self, pattern):
        """查询与给定模式匹配的事实"""
        subject, relation, object = pattern
        query_result = pyDatalog.ask(f"{relation}(X, Y)")
        results = []
        if query_result:
            for result in query_result:
                if (subject == 'X' or subject == result[0]) and \
                   (object == 'Y' or object == result[1]):
                    results.append((result[0], relation, result[1]))
        return results

    async def add_rule(self, condition, conclusion):
        """添加推理规则"""
        pyDatalog.load(f"{conclusion} <= {condition}")
        self.logger.debug(f"Added rule: {conclusion} <= {condition}")

    async def infer(self, context=None):
        """基于现有事实和规则进行推理"""
        inferred_facts = set()
        for fact in self.facts:
            subject, relation, object = fact
            # 基本推理规则
            if relation == 'present_in':
                # 如果A在场景中，且B与A相关，则B也可能在场景中
                related_objects = await self.query((subject, 'related_to', 'Y'))
                for _, _, related_obj in related_objects:
                    inferred_facts.add((related_obj, 'possibly_present', 'scene'))
            
            elif relation == 'mentioned_in':
                # 如果A在文本中被提到，且A暗示B，则B可能相关
                implied_objects = await self.query((subject, 'implies', 'Y'))
                for _, _, implied_obj in implied_objects:
                    inferred_facts.add((implied_obj, 'possibly_relevant', object))
        
        return inferred_facts

    async def explain(self, fact):
        """解释如何得出特定结论"""
        subject, relation, object = fact
        explanation = []
        
        # 检查直接事实
        if fact in self.facts:
            explanation.append(f"Direct fact: {subject} {relation} {object}")
            return explanation
        
        # 检查通过规则推导的事实
        inferred = await self.infer()
        if fact in inferred:
            explanation.append(f"Inferred fact: {subject} {relation} {object}")
            # 添加推导链
            supporting_facts = [f for f in self.facts if f[0] == subject or f[2] == object]
            for sf in supporting_facts:
                explanation.append(f"Supporting fact: {sf[0]} {sf[1]} {sf[2]}")
        
        return explanation if explanation else ["No explanation found"]

    async def validate(self, fact):
        """验证一个事实是否合理"""
        subject, relation, object = fact
        
        # 检查直接冲突
        contradictions = []
        for existing_fact in self.facts:
            if existing_fact[0] == subject and existing_fact[1] == relation and \
               existing_fact[2] != object:
                contradictions.append(existing_fact)
        
        if contradictions:
            return False, f"Contradicts existing facts: {contradictions}"
        
        # 检查一致性规则
        if relation == 'present_in':
            # 例如，检查对象是否可能出现在给定场景中
            scene_objects = await self.query(('X', 'valid_in', object))
            if scene_objects and (subject, 'valid_in', object) not in scene_objects:
                return False, f"{subject} is not valid in {object}"
        
        return True, "Fact is valid"

inference_engine = InferenceEngine()
