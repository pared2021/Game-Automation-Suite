import sqlite3
import aiosqlite
from utils.error_handler import log_exception
from utils.database_optimizer import optimize_database  # 引入通用方法
from utils.session_utils import save_game_session, get_session_stats  # 引入通用方法

class DatabaseManager:
    def __init__(self, db_path='game_data.db'):
        self.db_path = db_path

    @log_exception
    async def initialize(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS game_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time DATETIME,
                    end_time DATETIME,
                    duration INTEGER,
                    player_level INTEGER,
                    gold_earned INTEGER
                )
            ''')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_start_time ON game_sessions(start_time)')
            await db.commit()

    @log_exception
    async def save_game_session(self, start_time, end_time, duration, player_level, gold_earned):
        await save_game_session(self.db_path, start_time, end_time, duration, player_level, gold_earned)

    @log_exception
    async def get_session_stats(self, start_date=None, end_date=None):
        return await get_session_stats(self.db_path, start_date, end_date)

    @log_exception
    async def optimize_database(self):
        await optimize_database(self.db_path)  # 调用通用方法

database_manager = DatabaseManager()
