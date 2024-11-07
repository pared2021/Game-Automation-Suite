import statistics
import aiosqlite
from utils.session_utils import save_game_session, get_session_stats  # 引入通用方法

class DataHandler:
    def __init__(self, db_path='game_data.db'):
        self.db_path = db_path

    async def initialize(self):
        # 初始化数据库
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

    async def save_game_session(self, start_time, end_time, duration, player_level, gold_earned):
        # 使用通用方法保存游戏会话数据
        await save_game_session(self.db_path, start_time, end_time, duration, player_level, gold_earned)

    async def get_session_stats(self, start_date=None, end_date=None):
        # 使用通用方法获取会话统计数据
        return await get_session_stats(self.db_path, start_date, end_date)

    async def calculate_average_reward(self):
        # 计算平均奖励
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT AVG(gold_earned) FROM game_sessions") as cursor:
                result = await cursor.fetchone()
                return result[0] if result else 0

    async def calculate_win_rate(self):
        # 计算胜率
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM game_sessions WHERE player_level > 0") as cursor:
                total_games = await cursor.fetchone()
            async with db.execute("SELECT COUNT(*) FROM game_sessions WHERE player_level > 5") as cursor:
                won_games = await cursor.fetchone()
            return (won_games[0] / total_games[0]) * 100 if total_games[0] > 0 else 0

    async def get_performance_stats(self):
        # 获取性能统计数据
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT duration, player_level, gold_earned FROM game_sessions") as cursor:
                rows = await cursor.fetchall()
                durations = [row[0] for row in rows]
                levels = [row[1] for row in rows]
                gold_earned = [row[2] for row in rows]

        return {
            "avg_duration": statistics.mean(durations) if durations else 0,
            "avg_level": statistics.mean(levels) if levels else 0,
            "avg_gold": statistics.mean(gold_earned) if gold_earned else 0,
            "max_level": max(levels) if levels else 0,
            "max_gold": max(gold_earned) if gold_earned else 0
        }

    async def close(self):
        # 此方法可用于关闭任何打开的连接或执行清理操作
        pass
