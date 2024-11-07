from flask import Flask, render_template, jsonify, request
from game_automation.game_engine import GameEngine
from game_automation.rogue.rogue_manager import RogueManager
from utils.error_handler import GameAutomationError, InputError, DeviceError, OCRError, NetworkError

app = Flask(__name__)
game_engine = GameEngine('config/strategies.json')
rogue_manager = RogueManager()

@app.errorhandler(GameAutomationError)
def handle_game_automation_error(error):
    response = jsonify({"error": str(error)})
    response.status_code = 500
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/dashboard-data')
async def dashboard_data():
    try:
        current_task = game_engine.task_explorer.get_current_task()
        tasks = await game_engine.task_manager.get_task_status()
        exploration_progress = game_engine.task_explorer.get_exploration_progress()
        
        return jsonify({
            "current_task": current_task,
            "tasks": tasks,
            "exploration_progress": exploration_progress
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/start', methods=['POST'])
async def start_automation():
    try:
        await game_engine.initialize()
        await game_engine.run_game_loop()
        return jsonify({"status": "Automation started"})
    except GameAutomationError as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def stop_automation():
    game_engine.stop_automation = True
    return jsonify({"status": "Automation stopped"})

@app.route('/api/status')
async def get_status():
    try:
        game_state = await game_engine.get_game_state()
        return jsonify(game_state)
    except GameAutomationError as e:
        return jsonify({"error": str(e)}), 500

# ... (rest of the routes remain the same)

if __name__ == "__main__":
    app.run(debug=True)