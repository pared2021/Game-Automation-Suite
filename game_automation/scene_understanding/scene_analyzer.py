import cv2
import numpy as np
from game_automation.image_recognition import enhanced_image_recognition
from game_automation.nlp.language_processor import language_processor
from game_automation.reasoning.inference_engine import inference_engine
from utils.logger import detailed_logger
from utils.config_manager import config_manager

class SceneAnalyzer:
    def __init__(self):
        self.logger = detailed_logger
        self.config = config_manager.get('scene_understanding', {})
        self.preset_scenes = {
            'main_menu': {
                'key_elements': ['start_button', 'options_button', 'exit_button'],
                'potential_actions': ['start_game', 'open_options', 'exit_game']
            },
            'character_creation': {
                'key_elements': ['character_model', 'class_selection', 'attribute_points'],
                'potential_actions': ['select_class', 'customize_appearance', 'allocate_attributes']
            },
            'town': {
                'key_elements': ['npc', 'shop', 'quest_board', 'inn'],
                'potential_actions': ['talk_to_npc', 'enter_shop', 'check_quests', 'rest_at_inn']
            },
            'dungeon_entrance': {
                'key_elements': ['dungeon_gate', 'warning_sign', 'treasure_chest'],
                'potential_actions': ['enter_dungeon', 'read_sign', 'open_chest']
            },
            'battle': {
                'key_elements': ['player_character', 'enemy', 'health_bar', 'skill_buttons'],
                'potential_actions': ['attack', 'use_skill', 'use_item', 'flee']
            },
            'inventory': {
                'key_elements': ['item_slots', 'equipment_slots', 'gold_amount'],
                'potential_actions': ['equip_item', 'use_item', 'sort_inventory']
            },
            'skill_tree': {
                'key_elements': ['skill_nodes', 'skill_points', 'reset_button'],
                'potential_actions': ['learn_skill', 'upgrade_skill', 'reset_skills']
            },
            'quest_log': {
                'key_elements': ['active_quests', 'completed_quests', 'quest_details'],
                'potential_actions': ['view_quest_details', 'abandon_quest', 'claim_reward']
            },
            'world_map': {
                'key_elements': ['player_location', 'points_of_interest', 'fast_travel_points'],
                'potential_actions': ['set_waypoint', 'fast_travel', 'explore_area']
            },
            'dialogue': {
                'key_elements': ['npc_portrait', 'dialogue_text', 'response_options'],
                'potential_actions': ['choose_response', 'end_conversation', 'ask_question']
            }
        }

    async def analyze_game_scene(self, image, text):
        scene_objects = await enhanced_image_recognition.detect_objects(image)
        scene_text = await enhanced_image_recognition.recognize_text(image)
        text_analysis = await language_processor.analyze_text(text + " " + scene_text)
        
        # 将场景信息添加到推理引擎
        for obj in scene_objects:
            await inference_engine.add_fact(obj['name'], 'present_in', 'scene')
        
        for entity in text_analysis['entities']:
            await inference_engine.add_fact(entity[0], 'mentioned_in', 'text')
        
        # 进行场景理解
        understanding = await self.interpret_scene(scene_objects, text_analysis)
        self.logger.info(f"Scene understanding: {understanding}")
        return understanding

    async def interpret_scene(self, objects, text_analysis):
        scene_type = await self.determine_scene_type(objects, text_analysis)
        interpretation = {
            "scene_type": scene_type,
            "key_elements": await self.identify_key_elements(objects, text_analysis, scene_type),
            "potential_actions": await self.suggest_actions(objects, text_analysis, scene_type),
            "narrative_context": await self.extract_narrative_context(text_analysis)
        }
        return interpretation

    async def determine_scene_type(self, objects, text_analysis):
        object_types = [obj['name'] for obj in objects]
        text_content = ' '.join([entity[0] for entity in text_analysis['entities']])
        
        for scene_type, scene_data in self.preset_scenes.items():
            if all(element in object_types or element in text_content for element in scene_data['key_elements']):
                return scene_type
        
        # 如果没有匹配到预设场景，使用更通用的判断逻辑
        if 'enemy' in object_types or 'battle' in text_content:
            return 'battle'
        elif 'npc' in object_types or 'dialogue' in text_content:
            return 'dialogue'
        elif 'item' in object_types or 'inventory' in text_content:
            return 'inventory'
        else:
            return 'exploration'

    async def identify_key_elements(self, objects, text_analysis, scene_type):
        key_elements = set()
        
        # 添加预设场景的关键元素
        if scene_type in self.preset_scenes:
            key_elements.update(self.preset_scenes[scene_type]['key_elements'])
        
        # 添加检测到的对象
        for obj in objects:
            if obj['confidence'] > self.config.get('object_confidence_threshold', 0.8):
                key_elements.add(obj['name'])
        
        # 添加文本分析中的实体
        for entity in text_analysis['entities']:
            if entity[1] in ['PERSON', 'ORG', 'GPE', 'PRODUCT']:
                key_elements.add(entity[0])
        
        return list(key_elements)

    async def suggest_actions(self, objects, text_analysis, scene_type):
        actions = set()
        
        # 添加预设场景的潜在动作
        if scene_type in self.preset_scenes:
            actions.update(self.preset_scenes[scene_type]['potential_actions'])
        
        # 根据场景类型添加通用动作
        if scene_type == 'battle':
            actions.update(['attack', 'defend', 'use_item', 'use_skill'])
        elif scene_type == 'dialogue':
            actions.update(['continue_dialogue', 'end_conversation'])
        elif scene_type == 'inventory':
            actions.update(['equip_item', 'use_item', 'discard_item'])
        elif scene_type == 'exploration':
            actions.update(['move', 'interact', 'examine'])
        
        # 根据检测到的对象添加特定动作
        for obj in objects:
            if obj['name'] == 'door':
                actions.add('open_door')
            elif obj['name'] == 'chest':
                actions.add('open_chest')
            elif obj['name'] == 'npc':
                actions.add('talk_to_npc')
        
        # 根据文本分析添加动作
        for verb in text_analysis['verbs']:
            if verb in ['attack', 'fight', 'battle']:
                actions.add('initiate_combat')
            elif verb in ['buy', 'sell', 'trade']:
                actions.add('open_shop')
            elif verb in ['quest', 'mission', 'task']:
                actions.add('check_quest_log')
        
        return list(actions)

    async def extract_narrative_context(self, text_analysis):
        context = {
            'characters': [],
            'locations': [],
            'key_concepts': [],
            'time_references': [],
            'quest_related': []
        }
        
        for entity in text_analysis['entities']:
            if entity[1] == 'PERSON':
                context['characters'].append(entity[0])
            elif entity[1] == 'GPE' or entity[1] == 'LOC':
                context['locations'].append(entity[0])
            elif entity[1] == 'TIME' or entity[1] == 'DATE':
                context['time_references'].append(entity[0])
        
        context['key_concepts'] = text_analysis['noun_phrases'][:5]
        
        # 识别与任务相关的信息
        quest_keywords = ['quest', 'mission', 'task', 'objective']
        for sentence in text_analysis['sentences']:
            if any(keyword in sentence.lower() for keyword in quest_keywords):
                context['quest_related'].append(sentence)
        
        return context

    async def decompose_task(self, task_description):
        subtasks = await language_processor.analyze_text(task_description)
        decomposed_tasks = []
        for verb in subtasks['verbs']:
            related_nouns = [np for np in subtasks['noun_phrases'] if verb in np]
            if related_nouns:
                decomposed_tasks.append(f"{verb} {related_nouns[0]}")
            else:
                decomposed_tasks.append(verb)
        self.logger.info(f"Task decomposition: {decomposed_tasks}")
        return decomposed_tasks

scene_analyzer = SceneAnalyzer()