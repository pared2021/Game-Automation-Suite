import sqlite3
import aiosqlite
from utils.error_handler import log_exception

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
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT INTO game_sessions (start_time, end_time, duration, player_level, gold_earned)
                VALUES (?, ?, ?, ?, ?)
            ''', (start_time, end_time, duration, player_level, gold_earned))
            await db.commit()

    @log_exception
    async def get_session_stats(self, start_date=None, end_date=None):
        query = '''
            SELECT DATE(start_time) as date, COUNT(*) as session_count, 
                   AVG(duration) as avg_duration, AVG(player_level) as avg_level, 
                   SUM(gold_earned) as total_gold
            FROM game_sessions
        '''
        params = []
        if start_date:
            query += ' WHERE start_time >= ?'
            params.append(start_date)
        if end_date:
            query += ' AND start_time <= ?' if start_date else ' WHERE start_time <= ?'
            params.append(end_date)
        query += ' GROUP BY DATE(start_time)'

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                return await cursor.fetchall()

    @log_exception
    async def optimize_database(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("VACUUM")
            await db.execute("ANALYZE")

database_manager = DatabaseManager()