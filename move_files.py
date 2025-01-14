"""Move files for core system refactoring."""

import os
import shutil

def move_file(src: str, dst: str) -> None:
    """Move a file from src to dst."""
    try:
        # Create destination directory if it doesn't exist
        dst_dir = os.path.dirname(dst)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
            
        # Move the file
        shutil.move(src, dst)
        print(f"Moved {src} to {dst}")
    except Exception as e:
        print(f"Error moving {src} to {dst}: {e}")

def main():
    """Main function."""
    base_dir = "d:/UGit/Game-Automation-Suite/game_automation/core"
    
    # Event system files
    move_file(
        f"{base_dir}/async_io/events.py",
        f"{base_dir}/events/event_types.py"
    )
    move_file(
        f"{base_dir}/async_io/event_dispatcher.py",
        f"{base_dir}/events/event_dispatcher.py"
    )
    move_file(
        f"{base_dir}/async_io/filters.py",
        f"{base_dir}/events/filters/event_filters.py"
    )
    move_file(
        f"{base_dir}/async_io/tracing.py",
        f"{base_dir}/events/tracing/event_tracer.py"
    )
    
    # Task monitoring
    move_file(
        f"{base_dir}/task/task_monitor.py",
        f"{base_dir}/monitor/task_monitor.py"
    )
    
    # Logging
    move_file(
        f"{base_dir}/log_manager.py",
        f"{base_dir}/logging/log_manager.py"
    )
    
if __name__ == "__main__":
    main()
