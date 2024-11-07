import statistics
import aiosqlite

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
        # 保存游戏会话数据
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT INTO game_sessions (start_time, end_time, duration, player_level, gold_earned)
                VALUES (?, ?, ?, ?, ?)
            ''', (start_time, end_time, duration, player_level, gold_earned))
            await db.commit()

    async def get_session_stats(self, start_date=None, end_date=None):
        # 获取会话统计数据
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