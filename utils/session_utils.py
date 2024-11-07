import aiosqlite

async def save_game_session(db_path, start_time, end_time, duration, player_level, gold_earned):
    async with aiosqlite.connect(db_path) as db:
        await db.execute('''
            INSERT INTO game_sessions (start_time, end_time, duration, player_level, gold_earned)
            VALUES (?, ?, ?, ?, ?)
        ''', (start_time, end_time, duration, player_level, gold_earned))
        await db.commit()

async def get_session_stats(db_path, start_date=None, end_date=None):
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

    async with aiosqlite.connect(db_path) as db:
        async with db.execute(query, params) as cursor:
            return await cursor.fetchall()
