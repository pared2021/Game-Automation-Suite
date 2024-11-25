from flask import Flask, request, jsonify
from flask_babel import Babel
from game_automation.core.engine import GameEngine
import yaml
import os
import asyncio

app = Flask(__name__)
babel = Babel(app)

# Load configuration
def load_config():
    config_path = os.path.join('config', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# Initialize game engine
config = load_config()
game_engine = GameEngine(config)

# Internationalization messages
messages = {
    'en-US': {
        'general': {
            'home': 'Home',
            'about': 'About',
            'game': 'Game',
            'rogue': 'Rogue Mode'
        },
        'home': {
            'title': 'Game Automation Suite',
            'status': 'Status',
            'start': 'Start Automation',
            'stop': 'Stop Automation',
            'playerStats': 'Player Stats',
            'health': 'Health',
            'mana': 'Mana',
            'currentTask': 'Current Task',
            'progress': 'Progress'
        }
    },
    'zh-CN': {
        'general': {
            'home': '首页',
            'about': '关于',
            'game': '游戏',
            'rogue': '肉鸽模式'
        },
        'home': {
            'title': '游戏自动化套件',
            'status': '状态',
            'start': '启动自动化',
            'stop': '停止自动化',
            'playerStats': '玩家状态',
            'health': '生命值',
            'mana': '魔法值',
            'currentTask': '当前任务',
            'progress': '进度'
        }
    }
}

@app.route('/api/start', methods=['POST'])
async def start_automation():
    """Start automation endpoint"""
    try:
        await game_engine.initialize()
        await game_engine.start()
        return jsonify({
            'status': 'success',
            'message': '自动化已启动'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'启动失败: {str(e)}'
        }), 500

@app.route('/api/stop', methods=['POST'])
async def stop_automation():
    """Stop automation endpoint"""
    try:
        await game_engine.stop()
        return jsonify({
            'status': 'success',
            'message': '自动化已停止'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'停止失败: {str(e)}'
        }), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current status endpoint"""
    try:
        state = game_engine.get_current_state()
        return jsonify({
            'status': 'success',
            'data': {
                'running': game_engine.is_running,
                'game_state': state,
                'tasks': {
                    'total': len(game_engine.task_manager.tasks),
                    'pending': len(game_engine.task_manager.task_queue),
                    'running': len(game_engine.task_manager.running_tasks),
                    'completed': len(game_engine.task_manager.completed_tasks),
                    'failed': len(game_engine.task_manager.failed_tasks)
                }
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'获取状态失败: {str(e)}'
        }), 500

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    """Get task list endpoint"""
    try:
        tasks = [
            {
                'id': task.task_id,
                'name': task.name,
                'status': task.status.name,
                'progress': task.progress,
                'start_time': task.start_time.isoformat() if task.start_time else None,
                'end_time': task.end_time.isoformat() if task.end_time else None,
                'error': task.error_message
            }
            for task in game_engine.task_manager.tasks.values()
        ]
        return jsonify({
            'status': 'success',
            'data': tasks
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'获取任务列表失败: {str(e)}'
        }), 500

@app.route('/api/task/<task_id>', methods=['GET'])
def get_task(task_id):
    """Get specific task details endpoint"""
    try:
        task = game_engine.task_manager.get_task(task_id)
        if not task:
            return jsonify({
                'status': 'error',
                'message': '任务不存在'
            }), 404
            
        return jsonify({
            'status': 'success',
            'data': {
                'id': task.task_id,
                'name': task.name,
                'status': task.status.name,
                'progress': task.progress,
                'start_time': task.start_time.isoformat() if task.start_time else None,
                'end_time': task.end_time.isoformat() if task.end_time else None,
                'error': task.error_message,
                'metrics': task.performance_metrics
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'获取任务详情失败: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
