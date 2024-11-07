import aiosqlite
from error_handler import log_exception  # 更新导入路径

@log_exception
async def optimize_database(db_path='game_data.db'):
    async with aiosqlite.connect(db_path) as db:
        await db.execute("VACUUM")
        await db.execute("ANALYZE")
