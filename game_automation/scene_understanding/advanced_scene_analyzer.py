import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from game_automation.scene_understanding.yolov5 import YOLOv5
from utils.config_manager import config_manager
from utils.logger import detailed_logger

class AdvancedSceneAnalyzer:
    def __init__(self):
        self.logger = detailed_logger
        self.config = config_manager.get('scene_understanding', {})
        self.yolo_model = YOLOv5(self.config.get('yolo_model_path', 'yolov5s.pt'))
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])
        self.scene_types = {
            'battle': ['enemy', 'health_bar', 'skill_button'],
            'town': ['npc', 'shop', 'quest_board'],
            'inventory': ['item', 'equipment_slot', 'gold_amount'],
            'world_map': ['map_icon', 'player_location', 'quest_marker'],
            'dialogue': ['dialogue_box', 'npc_portrait', 'dialogue_options']
        }
        self.dynamic_object_library = {}
        self.history = []

    async def initialize(self):
        # Load any pre-trained models or additional resources
        pass

    async def analyze_scene(self, image, game_state):
        image_tensor = self.transform(Image.fromarray(image)).unsqueeze(0)
        results = self.yolo_model(image_tensor)
        objects = results.pandas().xyxy[0].to_dict(orient="records")
        
        scene_type = self.determine_scene_type(objects)
        context_analysis = self.analyze_context(objects, game_state, scene_type)
        self.update_dynamic_object_library(objects, scene_type)
        self.update_history(context_analysis)
        
        return {
            'scene_type': scene_type,
            'detected_objects': objects,
            'context_analysis': context_analysis,
            'dynamic_objects': self.get_dynamic_objects(scene_type),
            'long_term_analysis': self.analyze_long_term_trends()
        }

    def determine_scene_type(self, objects):
        detected_classes = set(obj['name'] for obj in objects)
        for scene_type, required_objects in self.scene_types.items():
            if all(obj in detected_classes for obj in required_objects):
                return scene_type
        return 'unknown'

    def analyze_context(self, objects, game_state, scene_type):
        context = {
            'player_status': self.analyze_player_status(game_state),
            'environmental_factors': self.analyze_environment(objects, scene_type),
            'potential_threats': self.identify_threats(objects, game_state),
            'available_actions': self.determine_available_actions(objects, scene_type, game_state)
        }
        return context

    def analyze_player_status(self, game_state):
        return {
            'health': game_state['player_stats']['health'],
            'mana': game_state['player_stats']['mana'],
            'level': game_state['player_stats']['level'],
            'experience': game_state['player_stats']['experience']
        }

    def analyze_environment(self, objects, scene_type):
        environment = {
            'obstacles': [obj for obj in objects if obj['name'] in ['wall', 'rock', 'tree']],
            'interactive_elements': [obj for obj in objects if obj['name'] in ['door', 'chest', 'lever']]
        }
        if scene_type == 'battle':
            environment['battle_elements'] = [obj for obj in objects if obj['name'] in ['enemy', 'boss', 'skill_effect']]
        elif scene_type == 'town':
            environment['npcs'] = [obj for obj in objects if obj['name'] == 'npc']
            environment['shops'] = [obj for obj in objects if obj['name'] == 'shop']
        return environment

    def identify_threats(self, objects, game_state):
        threats = [obj for obj in objects if obj['name'] in ['enemy', 'trap', 'boss']]
        return sorted(threats, key=lambda x: self.calculate_threat_level(x, game_state), reverse=True)

    def calculate_threat_level(self, threat, game_state):
        base_threat = {'enemy': 5, 'trap': 3, 'boss': 10}.get(threat['name'], 1)
        player_level = game_state['player_stats']['level']
        distance = ((threat['xmin'] + threat['xmax']) / 2 - game_state['player_position']['x'])**2 + \
                   ((threat['ymin'] + threat['ymax']) / 2 - game_state['player_position']['y'])**2
        return base_threat * (1 + threat.get('level', player_level) / player_level) / (distance + 1)

    def determine_available_actions(self, objects, scene_type, game_state):
        actions = ['move', 'interact']
        if scene_type == 'battle':
            actions.extend(['attack', 'use_skill', 'use_item', 'defend'])
        elif scene_type == 'town':
            actions.extend(['talk_to_npc', 'enter_shop', 'accept_quest'])
        elif scene_type == 'inventory':
            actions.extend(['equip_item', 'use_item', 'sort_inventory'])
        elif scene_type == 'world_map':
            actions.extend(['set_destination', 'fast_travel', 'view_quest_info'])
        elif scene_type == 'dialogue':
            actions.extend(['choose_dialogue_option', 'end_conversation'])
        return actions

    def update_dynamic_object_library(self, objects, scene_type):
        for obj in objects:
            if obj['name'] not in self.dynamic_object_library:
                self.dynamic_object_library[obj['name']] = {
                    'count': 1,
                    'scenes': {scene_type: 1},
                    'last_seen': 0
                }
            else:
                self.dynamic_object_library[obj['name']]['count'] += 1
                self.dynamic_object_library[obj['name']]['scenes'][scene_type] = \
                    self.dynamic_object_library[obj['name']]['scenes'].get(scene_type, 0) + 1
                self.dynamic_object_library[obj['name']]['last_seen'] = 0

        for obj_name in self.dynamic_object_library:
            if obj_name not in [obj['name'] for obj in objects]:
                self.dynamic_object_library[obj_name]['last_seen'] += 1

    def get_dynamic_objects(self, scene_type):
        return {
            obj_name: info for obj_name, info in self.dynamic_object_library.items()
            if scene_type in info['scenes'] and info['last_seen'] < self.config.get('dynamic_object_expiry', 100)
        }

    def update_history(self, context_analysis):
        self.history.append(context_analysis)
        if len(self.history) > self.config.get('max_history_length', 100):
            self.history.pop(0)

    def analyze_long_term_trends(self):
        if len(self.history) < 10:
            return "Not enough historical data"
        
        scene_types = [h['scene_type'] for h in self.history]
        most_common_scene = max(set(scene_types), key=scene_types.count)
        
        health_trend = [h['player_status']['health'] for h in self.history]
        health_trend = np.array(health_trend)
        health_slope = np.polyfit(range(len(health_trend)), health_trend, 1)[0]
        
        return {
            'dominant_scene_type': most_common_scene,
            'health_trend': 'improving' if health_slope > 0 else 'declining',
            'average_threats_per_scene': sum(len(h['potential_threats']) for h in self.history) / len(self.history)
        }

advanced_scene_analyzer = AdvancedSceneAnalyzer()