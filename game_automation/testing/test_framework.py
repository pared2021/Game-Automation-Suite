from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import asyncio
import json
import os
import yaml
import pytest
from enum import Enum, auto

from utils.error_handler import log_exception, GameAutomationError
from utils.logger import detailed_logger

class TestType(Enum):
    """测试类型枚举"""
    UNIT = auto()       # 单元测试
    INTEGRATION = auto() # 集成测试
    E2E = auto()        # 端到端测试
    PERFORMANCE = auto() # 性能测试
    BENCHMARK = auto()  # 基准测试

@dataclass
class TestCase:
    """测试用例数据类"""
    test_id: str
    name: str
    type: TestType
    description: str
    setup: Optional[Dict[str, Any]]
    inputs: Dict[str, Any]
    expected: Dict[str, Any]
    cleanup: Optional[Dict[str, Any]]
    timeout: float
    tags: List[str]

@dataclass
class TestResult:
    """测试结果数据类"""
    test_id: str
    success: bool
    error_message: Optional[str]
    duration: float
    timestamp: datetime
    metrics: Dict[str, Any]

class TestLoader:
    """测试加载器"""
    
    @staticmethod
    def load_test_suite(filepath: str) -> List[TestCase]:
        """加载测试套件
        
        Args:
            filepath: 测试套件文件路径
            
        Returns:
            List[TestCase]: 测试用例列表
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if filepath.endswith('.json'):
                    data = json.load(f)
                elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
                    data = yaml.safe_load(f)
                else:
                    raise GameAutomationError("不支持的文件格式")
                    
            test_cases = []
            for case_data in data['test_cases']:
                test_cases.append(TestCase(
                    test_id=case_data['id'],
                    name=case_data['name'],
                    type=TestType[case_data['type']],
                    description=case_data['description'],
                    setup=case_data.get('setup'),
                    inputs=case_data['inputs'],
                    expected=case_data['expected'],
                    cleanup=case_data.get('cleanup'),
                    timeout=case_data.get('timeout', 30.0),
                    tags=case_data.get('tags', [])
                ))
                
            return test_cases
            
        except Exception as e:
            detailed_logger.error(f"加载测试套件失败: {str(e)}")
            return []

class TestRunner:
    """测试运行器"""
    
    def __init__(self):
        """初始化测试运行器"""
        self.results: Dict[str, TestResult] = {}
        self.current_test: Optional[TestCase] = None
        self.setup_fixtures: Dict[str, Callable] = {}
        self.cleanup_fixtures: Dict[str, Callable] = {}

    def register_setup(self, name: str, fixture: Callable) -> None:
        """注册设置函数
        
        Args:
            name: 函数名称
            fixture: 设置函数
        """
        self.setup_fixtures[name] = fixture

    def register_cleanup(self, name: str, fixture: Callable) -> None:
        """注册清理函数
        
        Args:
            name: 函数名称
            fixture: 清理函数
        """
        self.cleanup_fixtures[name] = fixture

    async def run_test(self, test_case: TestCase) -> TestResult:
        """运行测试用例
        
        Args:
            test_case: 测试用例
            
        Returns:
            TestResult: 测试结果
        """
        self.current_test = test_case
        start_time = datetime.now()
        
        try:
            # 运行设置
            if test_case.setup:
                await self._run_setup(test_case.setup)
                
            # 运行测试
            result = await asyncio.wait_for(
                self._run_test_case(test_case),
                timeout=test_case.timeout
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            test_result = TestResult(
                test_id=test_case.test_id,
                success=result['success'],
                error_message=result.get('error'),
                duration=duration,
                timestamp=datetime.now(),
                metrics=result.get('metrics', {})
            )
            
        except asyncio.TimeoutError:
            duration = (datetime.now() - start_time).total_seconds()
            test_result = TestResult(
                test_id=test_case.test_id,
                success=False,
                error_message="测试超时",
                duration=duration,
                timestamp=datetime.now(),
                metrics={}
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            test_result = TestResult(
                test_id=test_case.test_id,
                success=False,
                error_message=str(e),
                duration=duration,
                timestamp=datetime.now(),
                metrics={}
            )
            
        finally:
            # 运行清理
            if test_case.cleanup:
                await self._run_cleanup(test_case.cleanup)
                
            self.current_test = None
            self.results[test_case.test_id] = test_result
            
        return test_result

    async def _run_setup(self, setup: Dict[str, Any]) -> None:
        """运行设置函数
        
        Args:
            setup: 设置配置
        """
        for name, params in setup.items():
            if name in self.setup_fixtures:
                try:
                    await self.setup_fixtures[name](**params)
                except Exception as e:
                    raise GameAutomationError(f"设置失败 {name}: {str(e)}")

    async def _run_cleanup(self, cleanup: Dict[str, Any]) -> None:
        """运行清理函数
        
        Args:
            cleanup: 清理配置
        """
        for name, params in cleanup.items():
            if name in self.cleanup_fixtures:
                try:
                    await self.cleanup_fixtures[name](**params)
                except Exception as e:
                    detailed_logger.error(f"清理失败 {name}: {str(e)}")

    async def _run_test_case(self, test_case: TestCase) -> Dict[str, Any]:
        """运行测试用例
        
        Args:
            test_case: 测试用例
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        raise NotImplementedError("TestRunner._run_test_case() must be implemented by subclass")

class AutomationTestRunner(TestRunner):
    """自动化测试运行器"""
    
    def __init__(self, game_engine):
        """初始化自动化测试运行器
        
        Args:
            game_engine: 游戏引擎实例
        """
        super().__init__()
        self.game_engine = game_engine

    async def _run_test_case(self, test_case: TestCase) -> Dict[str, Any]:
        """运行测试用例
        
        Args:
            test_case: 测试用例
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        try:
            # 准备输入
            context = {
                'game_engine': self.game_engine,
                'inputs': test_case.inputs
            }
            
            # 运行测试逻辑
            if test_case.type == TestType.UNIT:
                result = await self._run_unit_test(context)
            elif test_case.type == TestType.INTEGRATION:
                result = await self._run_integration_test(context)
            elif test_case.type == TestType.E2E:
                result = await self._run_e2e_test(context)
            elif test_case.type == TestType.PERFORMANCE:
                result = await self._run_performance_test(context)
            elif test_case.type == TestType.BENCHMARK:
                result = await self._run_benchmark_test(context)
            else:
                raise GameAutomationError(f"不支持的测试类型: {test_case.type}")
                
            # 验证结果
            success = self._verify_result(result, test_case.expected)
            
            return {
                'success': success,
                'metrics': result.get('metrics', {})
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    async def _run_unit_test(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """运行单元测试
        
        Args:
            context: 测试上下文
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        # 获取要测试的函数
        func_name = context['inputs']['function']
        func = getattr(self.game_engine, func_name)
        
        # 运行函数
        result = await func(**context['inputs'].get('params', {}))
        
        return {
            'result': result
        }

    async def _run_integration_test(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """运行集成测试
        
        Args:
            context: 测试上下文
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        # 获取测试场景
        scenario = context['inputs']['scenario']
        
        # 运行场景
        metrics = {}
        for step in scenario:
            # 执行步骤
            func_name = step['function']
            func = getattr(self.game_engine, func_name)
            result = await func(**step.get('params', {}))
            
            # 收集指标
            if 'metrics' in result:
                metrics.update(result['metrics'])
                
        return {
            'metrics': metrics
        }

    async def _run_e2e_test(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """运行端到端测试
        
        Args:
            context: 测试上下文
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        # 获取测试流程
        workflow = context['inputs']['workflow']
        
        # 运行工作流
        metrics = {}
        for action in workflow:
            # 执行动作
            func_name = action['action']
            func = getattr(self.game_engine, func_name)
            result = await func(**action.get('params', {}))
            
            # 验证结果
            if 'expect' in action:
                success = self._verify_result(result, action['expect'])
                if not success:
                    return {
                        'error': f"动作验证失败: {func_name}"
                    }
                    
            # 收集指标
            if 'metrics' in result:
                metrics.update(result['metrics'])
                
        return {
            'metrics': metrics
        }

    async def _run_performance_test(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """运行性能测试
        
        Args:
            context: 测试上下文
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        # 获取测试配置
        config = context['inputs']
        iterations = config.get('iterations', 100)
        
        # 运行测试
        metrics = {
            'response_times': [],
            'error_count': 0
        }
        
        for _ in range(iterations):
            start_time = time.time()
            try:
                # 执行操作
                func_name = config['function']
                func = getattr(self.game_engine, func_name)
                await func(**config.get('params', {}))
                
                # 记录响应时间
                response_time = time.time() - start_time
                metrics['response_times'].append(response_time)
                
            except Exception:
                metrics['error_count'] += 1
                
        # 计算统计指标
        response_times = metrics['response_times']
        if response_times:
            metrics.update({
                'min_time': min(response_times),
                'max_time': max(response_times),
                'avg_time': sum(response_times) / len(response_times),
                'p95_time': np.percentile(response_times, 95),
                'p99_time': np.percentile(response_times, 99)
            })
            
        return {
            'metrics': metrics
        }

    async def _run_benchmark_test(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """运行基准测试
        
        Args:
            context: 测试上下文
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        # 获取测试配置
        config = context['inputs']
        duration = config.get('duration', 60)  # 默认运行60秒
        
        # 运行测试
        metrics = {
            'operations': 0,
            'errors': 0,
            'response_times': []
        }
        
        end_time = time.time() + duration
        while time.time() < end_time:
            start_time = time.time()
            try:
                # 执行操作
                func_name = config['function']
                func = getattr(self.game_engine, func_name)
                await func(**config.get('params', {}))
                
                # 更新指标
                metrics['operations'] += 1
                response_time = time.time() - start_time
                metrics['response_times'].append(response_time)
                
            except Exception:
                metrics['errors'] += 1
                
        # 计算统计指标
        response_times = metrics['response_times']
        if response_times:
            metrics.update({
                'min_time': min(response_times),
                'max_time': max(response_times),
                'avg_time': sum(response_times) / len(response_times),
                'p95_time': np.percentile(response_times, 95),
                'p99_time': np.percentile(response_times, 99),
                'throughput': metrics['operations'] / duration,
                'error_rate': metrics['errors'] / metrics['operations']
            })
            
        return {
            'metrics': metrics
        }

    def _verify_result(self, actual: Any, expected: Any) -> bool:
        """验证测试结果
        
        Args:
            actual: 实际结果
            expected: 期望结果
            
        Returns:
            bool: 是否匹配
        """
        # 处理字典类型
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                return False
            return all(
                key in actual and self._verify_result(actual[key], value)
                for key, value in expected.items()
            )
            
        # 处理列表类型
        elif isinstance(expected, list):
            if not isinstance(actual, list):
                return False
            if len(expected) != len(actual):
                return False
            return all(
                self._verify_result(a, e)
                for a, e in zip(actual, expected)
            )
            
        # 处理基本类型
        else:
            return actual == expected

class TestReporter:
    """测试报告生成器"""
    
    def __init__(self, report_dir: str = "reports"):
        """初始化报告生成器
        
        Args:
            report_dir: 报告目录
        """
        self.report_dir = report_dir
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)

    def generate_report(self, results: Dict[str, TestResult],
                       report_format: str = 'html') -> str:
        """生成测试报告
        
        Args:
            results: 测试结果
            report_format: 报告格式
            
        Returns:
            str: 报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if report_format == 'html':
            return self._generate_html_report(results, timestamp)
        elif report_format == 'json':
            return self._generate_json_report(results, timestamp)
        else:
            raise GameAutomationError(f"不支持的报告格式: {report_format}")

    def _generate_html_report(self, results: Dict[str, TestResult],
                            timestamp: str) -> str:
        """生成HTML报告
        
        Args:
            results: 测试结果
            timestamp: 时间戳
            
        Returns:
            str: 报告文件路径
        """
        # 计算统计信息
        stats = self._calculate_statistics(results)
        
        # 生成HTML内容
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Report - {timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ margin-bottom: 20px; }}
                .test-case {{ margin-bottom: 10px; padding: 10px; border: 1px solid #ddd; }}
                .success {{ background-color: #dff0d8; }}
                .failure {{ background-color: #f2dede; }}
            </style>
        </head>
        <body>
            <h1>Test Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Tests: {stats['total']}</p>
                <p>Passed: {stats['passed']} ({stats['pass_rate']:.1f}%)</p>
                <p>Failed: {stats['failed']}</p>
                <p>Average Duration: {stats['avg_duration']:.2f}s</p>
            </div>
            <div class="test-cases">
                <h2>Test Cases</h2>
        """
        
        # 添加测试用例详情
        for test_id, result in results.items():
            status_class = "success" if result.success else "failure"
            html_content += f"""
                <div class="test-case {status_class}">
                    <h3>{test_id}</h3>
                    <p>Status: {'Passed' if result.success else 'Failed'}</p>
                    <p>Duration: {result.duration:.2f}s</p>
            """
            
            if result.error_message:
                html_content += f"<p>Error: {result.error_message}</p>"
                
            if result.metrics:
                html_content += "<h4>Metrics</h4><ul>"
                for key, value in result.metrics.items():
                    html_content += f"<li>{key}: {value}</li>"
                html_content += "</ul>"
                
            html_content += "</div>"
            
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # 保存报告
        report_path = os.path.join(
            self.report_dir,
            f"test_report_{timestamp}.html"
        )
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return report_path

    def _generate_json_report(self, results: Dict[str, TestResult],
                            timestamp: str) -> str:
        """生成JSON报告
        
        Args:
            results: 测试结果
            timestamp: 时间戳
            
        Returns:
            str: 报告文件路径
        """
        # 转换结果为JSON格式
        report_data = {
            'timestamp': timestamp,
            'statistics': self._calculate_statistics(results),
            'results': {
                test_id: {
                    'success': result.success,
                    'error_message': result.error_message,
                    'duration': result.duration,
                    'timestamp': result.timestamp.isoformat(),
                    'metrics': result.metrics
                }
                for test_id, result in results.items()
            }
        }
        
        # 保存报告
        report_path = os.path.join(
            self.report_dir,
            f"test_report_{timestamp}.json"
        )
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
            
        return report_path

    def _calculate_statistics(self, results: Dict[str, TestResult]) -> Dict[str, Any]:
        """计算测试统计信息
        
        Args:
            results: 测试结果
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        total = len(results)
        passed = sum(1 for result in results.values() if result.success)
        failed = total - passed
        
        durations = [result.duration for result in results.values()]
        avg_duration = sum(durations) / total if total > 0 else 0
        
        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': (passed / total * 100) if total > 0 else 0,
            'avg_duration': avg_duration
        }
