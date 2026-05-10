# agent_tools.py
import pandas as pd
import numpy as np
import json
import re
import traceback
import threading
import time
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

# === Конфигурация песочницы ===
MAX_EXECUTION_TIME = 10  # секунд
MAX_OUTPUT_LENGTH = 5000  # символов в ответе инструмента


def _run_with_timeout(func, args=(), kwargs={}, timeout=MAX_EXECUTION_TIME):
    """Кроссплатформенное выполнение функции с таймаутом"""
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
            
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        raise TimeoutError(f"Выполнение кода превысило лимит в {timeout} сек. Возможна бесконечный цикл.")
    if exception[0]:
        raise exception[0]
    return result[0]


@contextmanager
def safe_execution_environment():
    """Контекстный менеджер для безопасного выполнения кода"""
    safe_builtins = {
        'len': len, 'range': range, 'str': str, 'int': int, 
        'float': float, 'list': list, 'dict': dict, 'set': set,
        'sum': sum, 'min': min, 'max': max, 'abs': abs,
        'round': round, 'sorted': sorted, 'enumerate': enumerate,
        'zip': zip, 'map': map, 'filter': filter,
        'True': True, 'False': False, 'None': None,
        '__name__': 'safe_sandbox'
    }
    yield {'__builtins__': safe_builtins}


def sanitize_code(code: str) -> str:
    """Очистка кода от потенциально опасных конструкций"""
    dangerous_patterns = [
        r'__import__', r'exec\s*\(', r'eval\s*\(', r'compile\s*\(',
        r'open\s*\(', r'os\.', r'sys\.', r'subprocess', r'pickle',
        r'import\s+os', r'import\s+sys', r'import\s+subprocess',
        r'__class__', r'__mro__', r'__subclasses__', r'gc\.', r'ctypes'
    ]
    for pattern in dangerous_patterns:
        code = re.sub(pattern, '# [BLOCKED]', code, flags=re.IGNORECASE)
    return code


def execute_pandas_code(code: str, df: pd.DataFrame, df_name: str = "df") -> Dict[str, Any]:
    """Безопасное выполнение pandas-кода в песочнице с таймаутом"""
    start_time = time.time()
    
    try:
        code = sanitize_code(code)
        
        def _run_sandbox():
            with safe_execution_environment() as safe_globals:
                import pandas as _pd, numpy as _np, math, statistics, json, re
                safe_globals.update({
                    'pd': _pd, 'np': _np, 'math': math, 
                    'statistics': statistics, 'json': json, 're': re,
                    df_name: df.copy()
                })
                
                output_buffer = []
                def safe_print(*args, **kwargs):
                    output_buffer.append(' '.join(map(str, args)))
                safe_globals['print'] = safe_print
                
                exec(code, safe_globals, safe_globals)
                
                # Приоритет 1: явная переменная result
                if 'result' in safe_globals:
                    return safe_globals['result'], output_buffer
                # Приоритет 2: вывод через print()
                if output_buffer:
                    return '\n'.join(output_buffer), output_buffer
                return None, output_buffer

        # Выполняем с контролем времени
        result, output_buffer = _run_with_timeout(_run_sandbox, timeout=MAX_EXECUTION_TIME)
        
        # Форматирование вывода
        if isinstance(result, pd.DataFrame):
            output = {
                'type': 'dataframe',
                'shape': result.shape,
                'columns': result.columns.tolist(),
                'head': result.head(10).to_dict(orient='records'),
                'dtypes': result.dtypes.astype(str).to_dict()
            }
        elif isinstance(result, pd.Series):
            output = {
                'type': 'series',
                'name': str(result.name),
                'head': result.head(20).to_dict(),
                'dtype': str(result.dtype)
            }
        elif isinstance(result, (dict, list, str, int, float, bool, type(None))):
            val_str = str(result)
            output = {
                'type': type(result).__name__,
                'value': val_str if len(val_str) < MAX_OUTPUT_LENGTH else val_str[:MAX_OUTPUT_LENGTH] + '...[truncated]'
            }
        else:
            output = {'type': 'unknown', 'value': str(result)[:MAX_OUTPUT_LENGTH]}
            
        if output_buffer:
            output['printed'] = '\n'.join(output_buffer)
            
        return {
            'success': True, 
            'result': output, 
            'execution_time': round(time.time() - start_time, 3)
        }
            
    except TimeoutError as e:
        return {'success': False, 'error': str(e), 'execution_time': round(time.time() - start_time, 3)}
    except Exception as e:
        return {
            'success': False, 
            'error': f"{type(e).__name__}: {str(e)}",
            'traceback': traceback.format_exc()[:800],
            'execution_time': round(time.time() - start_time, 3)
        }


# === Реестр инструментов ===
class ToolRegistry:
    """Реестр доступных инструментов для агента"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.tools = {
            "execute_python": {
                "name": "execute_python",
                "description": "Выполнить pandas/Python код для анализа данных. "
                              "Используйте переменную 'df' для доступа к данным. "
                              "Присвойте итог переменной 'result' или выведите через print().",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python-код. Пример: 'result = df.groupby(\"category\")[\"sales\"].mean().round(2)'"
                        }
                    },
                    "required": ["code"]
                },
                "function": lambda code: execute_pandas_code(code, self.df)
            },
            "get_column_info": {
                "name": "get_column_info", 
                "description": "Получить информацию о конкретной колонке: тип, уникальные значения, статистики",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column": {"type": "string", "description": "Имя колонки"},
                        "stat": {
                            "type": "string", 
                            "enum": ["describe", "unique", "nulls", "dtype", "all"],
                            "description": "Тип запрашиваемой информации"
                        }
                    },
                    "required": ["column", "stat"]
                },
                "function": lambda column, stat="all": self._get_column_info(column, stat)
            },
            "filter_data": {
                "name": "filter_data",
                "description": "Отфильтровать DataFrame по условию и вернуть первые N строк",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "condition": {
                            "type": "string",
                            "description": "Условие фильтрации. Пример: 'sales > 1000' или 'category == \"Electronics\"'"
                        },
                        "limit": {"type": "integer", "description": "Максимум строк для возврата", "default": 10}
                    },
                    "required": ["condition"]
                },
                "function": lambda condition, limit=10: self._filter_data(condition, limit)
            }
        }
    
    def _get_column_info(self, column: str, stat: str) -> Dict[str, Any]:
        if column not in self.df.columns:
            return {'success': False, 'error': f"Колонка '{column}' не найдена. Доступные: {list(self.df.columns)}"}
        
        col = self.df[column]
        result = {}
        
        if stat in ["all", "dtype"]:
            result['dtype'] = str(col.dtype)
        if stat in ["all", "describe"] and pd.api.types.is_numeric_dtype(col):
            result['describe'] = col.describe().to_dict()
        if stat in ["all", "unique"]:
            result['unique_count'] = col.nunique()
            result['sample_unique'] = col.dropna().unique()[:10].tolist()
        if stat in ["all", "nulls"]:
            result['null_count'] = int(col.isna().sum())
            result['null_percentage'] = round(col.isna().mean() * 100, 2)
            
        return {'success': True, 'result': result}
    
    def _filter_data(self, condition: str, limit: int) -> Dict[str, Any]:
        """Безопасная фильтрация через pandas query вместо eval"""
        try:
            # sanitize убирает опасные импорты, но query() дополнительно изолирует от доступа к __builtins__
            safe_condition = sanitize_code(condition)
            filtered = self.df.query(safe_condition)
            if len(filtered) == 0:
                return {'success': True, 'result': {'type': 'info', 'value': 'Условие не вернуло строк. Проверьте синтаксис или данные.'}}
            return {
                'success': True,
                'result': {
                    'type': 'dataframe',
                    'shape': filtered.shape,
                    'head': filtered.head(limit).to_dict(orient='records')
                }
            }
        except Exception as e:
            return {'success': False, 'error': f"Ошибка фильтрации: {str(e)}"}
    
    def get_tool_schema(self) -> List[Dict]:
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"]
            }
            for t in self.tools.values()
        ]
    
    def execute(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        if tool_name not in self.tools:
            return {'success': False, 'error': f"Инструмент '{tool_name}' не найден"}
        try:
            return self.tools[tool_name]["function"](**kwargs)
        except Exception as e:
            return {'success': False, 'error': f"Ошибка выполнения: {str(e)}"}