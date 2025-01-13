from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import os
import time
from dataclasses import dataclass

from utils.error_handler import log_exception, GameAutomationError
from utils.logger import detailed_logger

@dataclass
class TaskHistoryEntry:
    """任务历史记录条目"""
    task_id: str
    task_name: str
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[timedelta]
    error_message: Optional[str]
    performance_metrics: Dict[str, Any]

class TaskHistory:
    """任务历史记录管理器"""

    def __init__(self, history_dir: str = "data/task_history"):
        """初始化历史记录管理器
        
        Args:
            history_dir: 历史记录保存目录
        """
        self.history_dir = history_dir
        self.current_history: List[TaskHistoryEntry] = []
        self.max_entries = 10000  # 最大记录数
        self.cleanup_interval = 24 * 3600  # 清理间隔(秒)
        self.last_cleanup_time = 0
        self._ensure_history_dir()

    def _ensure_history_dir(self) -> None:
        """确保历史记录目录存在"""
        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)
            detailed_logger.info(f"创建历史记录目录: {self.history_dir}")

    def _check_cleanup_needed(self) -> bool:
        """检查是否需要清理
        
        Returns:
            bool: 是否需要清理
        """
        current_time = time.time()
        
        # 检查时间间隔
        if current_time - self.last_cleanup_time > self.cleanup_interval:
            return True
            
        # 检查记录数量
        total_entries = 0
        for date_dir in os.listdir(self.history_dir):
            date_path = os.path.join(self.history_dir, date_dir)
            if os.path.isdir(date_path):
                total_entries += len(os.listdir(date_path))
                
        return total_entries >= self.max_entries

    @log_exception
    def add_entry(self, entry: TaskHistoryEntry) -> None:
        """添加历史记录
        
        Args:
            entry: 历史记录条目
        """
        # 检查是否需要清理
        if self._check_cleanup_needed():
            self.cleanup_old_records()
            self.last_cleanup_time = time.time()
        
        self.current_history.append(entry)
        self._save_entry(entry)
        detailed_logger.info(f"添加任务历史记录: {entry.task_name} ({entry.task_id})")
        
        # 限制内存中的记录数量
        if len(self.current_history) > self.max_entries:
            self.current_history = self.current_history[-self.max_entries:]

    def _save_entry(self, entry: TaskHistoryEntry) -> None:
        """保存单个历史记录
        
        Args:
            entry: 历史记录条目
        """
        try:
            # 创建日期目录
            date_str = entry.start_time.strftime("%Y-%m-%d")
            date_dir = os.path.join(self.history_dir, date_str)
            if not os.path.exists(date_dir):
                os.makedirs(date_dir)
            
            # 保存记录
            filename = f"{entry.task_id}_{entry.start_time.strftime('%H-%M-%S')}.json"
            filepath = os.path.join(date_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self._entry_to_dict(entry), f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            detailed_logger.error(f"保存任务历史记录失败: {str(e)}")

    def _entry_to_dict(self, entry: TaskHistoryEntry) -> Dict:
        """将历史记录条目转换为字典
        
        Args:
            entry: 历史记录条目
            
        Returns:
            Dict: 历史记录字典
        """
        return {
            'task_id': entry.task_id,
            'task_name': entry.task_name,
            'status': entry.status,
            'start_time': entry.start_time.isoformat(),
            'end_time': entry.end_time.isoformat() if entry.end_time else None,
            'duration': str(entry.duration) if entry.duration else None,
            'error_message': entry.error_message,
            'performance_metrics': entry.performance_metrics
        }

    @log_exception
    def get_task_history(self, task_id: str, 
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> List[TaskHistoryEntry]:
        """获取指定任务的历史记录
        
        Args:
            task_id: 任务ID
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            List[TaskHistoryEntry]: 历史记录列表
        """
        history = []
        
        try:
            # 设置日期范围
            if not start_date:
                start_date = datetime.now() - timedelta(days=7)  # 默认查询最近7天
            if not end_date:
                end_date = datetime.now()
            
            # 遍历日期目录
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                date_dir = os.path.join(self.history_dir, date_str)
                
                if os.path.exists(date_dir):
                    # 查找匹配的记录文件
                    for filename in os.listdir(date_dir):
                        if filename.startswith(f"{task_id}_"):
                            filepath = os.path.join(date_dir, filename)
                            entry = self._load_entry(filepath)
                            if entry:
                                history.append(entry)
                
                current_date += timedelta(days=1)
                
        except Exception as e:
            detailed_logger.error(f"获取任务历史记录失败: {str(e)}")
            
        return history

    def _load_entry(self, filepath: str) -> Optional[TaskHistoryEntry]:
        """加载单个历史记录
        
        Args:
            filepath: 记录文件路径
            
        Returns:
            Optional[TaskHistoryEntry]: 历史记录条目，加载失败返回None
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            return TaskHistoryEntry(
                task_id=data['task_id'],
                task_name=data['task_name'],
                status=data['status'],
                start_time=datetime.fromisoformat(data['start_time']),
                end_time=datetime.fromisoformat(data['end_time']) if data['end_time'] else None,
                duration=timedelta(seconds=float(data['duration'].split(':')[-1])) if data['duration'] else None,
                error_message=data['error_message'],
                performance_metrics=data['performance_metrics']
            )
            
        except Exception as e:
            detailed_logger.error(f"加载任务历史记录失败 {filepath}: {str(e)}")
            return None

    @log_exception
    def get_statistics(self, start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """获取历史统计信息
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'average_duration': timedelta(),
            'total_duration': timedelta(),
            'success_rate': 0.0,
            'task_distribution': {},
            'error_distribution': {}
        }
        
        try:
            # 设置日期范围
            if not start_date:
                start_date = datetime.now() - timedelta(days=7)
            if not end_date:
                end_date = datetime.now()
            
            # 遍历所有记录
            durations = []
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                date_dir = os.path.join(self.history_dir, date_str)
                
                if os.path.exists(date_dir):
                    for filename in os.listdir(date_dir):
                        entry = self._load_entry(os.path.join(date_dir, filename))
                        if entry:
                            stats['total_tasks'] += 1
                            
                            # 统计成功/失败
                            if entry.status == 'COMPLETED':
                                stats['successful_tasks'] += 1
                            elif entry.status == 'FAILED':
                                stats['failed_tasks'] += 1
                                if entry.error_message:
                                    stats['error_distribution'][entry.error_message] = \
                                        stats['error_distribution'].get(entry.error_message, 0) + 1
                            
                            # 统计持续时间
                            if entry.duration:
                                durations.append(entry.duration)
                                stats['total_duration'] += entry.duration
                            
                            # 统计任务分布
                            stats['task_distribution'][entry.task_name] = \
                                stats['task_distribution'].get(entry.task_name, 0) + 1
                
                current_date += timedelta(days=1)
            
            # 计算平均持续时间和成功率
            if durations:
                stats['average_duration'] = stats['total_duration'] / len(durations)
            if stats['total_tasks'] > 0:
                stats['success_rate'] = stats['successful_tasks'] / stats['total_tasks']
            
        except Exception as e:
            detailed_logger.error(f"获取历史统计信息失败: {str(e)}")
            
        return stats

    @log_exception
    def cleanup_old_records(self, days: int = 30) -> None:
        """清理旧记录
        
        Args:
            days: 保留天数
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # 遍历日期目录
            for date_str in os.listdir(self.history_dir):
                try:
                    date = datetime.strptime(date_str, "%Y-%m-%d")
                    if date < cutoff_date:
                        date_dir = os.path.join(self.history_dir, date_str)
                        # 删除目录及其内容
                        for filename in os.listdir(date_dir):
                            os.remove(os.path.join(date_dir, filename))
                        os.rmdir(date_dir)
                        detailed_logger.info(f"清理历史记录: {date_str}")
                except ValueError:
                    continue  # 跳过无效的日期目录名
                    
        except Exception as e:
            detailed_logger.error(f"清理历史记录失败: {str(e)}")

    def get_performance_trends(self, task_id: str,
                             metric_name: str,
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None) -> Dict[str, List[float]]:
        """获取性能指标趋势
        
        Args:
            task_id: 任务ID
            metric_name: 指标名称
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict[str, List[float]]: 趋势数据
        """
        trends = {
            'timestamps': [],
            'values': []
        }
        
        try:
            history = self.get_task_history(task_id, start_date, end_date)
            for entry in history:
                if metric_name in entry.performance_metrics:
                    trends['timestamps'].append(entry.start_time.isoformat())
                    trends['values'].append(entry.performance_metrics[metric_name])
                    
        except Exception as e:
            detailed_logger.error(f"获取性能趋势失败: {str(e)}")
            
        return trends
