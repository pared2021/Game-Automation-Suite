import sqlite3
import os
from utils.logger import setup_logger
from utils.error_handler import log_exception
from utils.database_optimizer import optimize_database  # 引入通用方法

class DatabaseChecker:
    def __init__(self, db_path='game_data.db'):
        self.db_path = db_path
        self.logger = setup_logger()

    @log_exception
    def run_integrity_check(self):
        self.logger.info("Running database integrity check...")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            if result[0] == "ok":
                self.logger.info("Database integrity check passed.")
            else:
                self.logger.error(f"Database integrity check failed: {result[0]}")

    @log_exception
    def optimize_database(self):
        self.logger.info("Optimizing database...")
        # 调用通用方法
        import asyncio
        asyncio.run(optimize_database(self.db_path))
        self.logger.info("Database optimization completed.")

    @log_exception
    def check_table_structure(self):
        self.logger.info("Checking table structure...")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            for table in tables:
                self.logger.info(f"Checking structure of table: {table[0]}")
                cursor.execute(f"PRAGMA table_info({table[0]})")
                columns = cursor.fetchall()
                for column in columns:
                    self.logger.info(f"  Column: {column[1]}, Type: {column[2]}")

    @log_exception
    def run_basic_queries(self):
        self.logger.info("Running basic queries to check data consistency...")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM game_sessions")
            session_count = cursor.fetchone()[0]
            self.logger.info(f"Total game sessions: {session_count}")

            cursor.execute("SELECT AVG(duration) FROM game_sessions")
            avg_duration = cursor.fetchone()[0]
            self.logger.info(f"Average session duration: {avg_duration:.2f} seconds")

    @log_exception
    def check_indexes(self):
        self.logger.info("Checking database indexes...")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = cursor.fetchall()
            for index in indexes:
                self.logger.info(f"Found index: {index[0]}")

    @log_exception
    def run_full_check(self):
        self.run_integrity_check()
        self.optimize_database()
        self.check_table_structure()
        self.run_basic_queries()
        self.check_indexes()
        self.logger.info("Full database check completed.")

if __name__ == "__main__":
    checker = DatabaseChecker()
    checker.run_full_check()
