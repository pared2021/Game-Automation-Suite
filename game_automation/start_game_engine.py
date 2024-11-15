import asyncio
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from game_automation.core.engine.game_engine import GameEngine

async def main():
    print("Initializing game engine...")
    engine = GameEngine()
    await engine.initialize()
    
    try:
        print("Game engine started. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down the game engine...")
    finally:
        await engine.cleanup()
        print("Game engine shutdown complete.")

if __name__ == "__main__":
    asyncio.run(main())
