# agent_loop.py
import json
import re
import time
import functools
from typing import List, Dict, Optional, Tuple, Any
import requests
import uuid

from agent_tools import ToolRegistry


class DataAnalysisAgent:
    """Агент для анализа данных с циклом: запрос → инструмент → результат → интерпретация"""
    
    MAX_ITERATIONS = 5
    TEMPERATURE = 0.2
    TOKEN_TTL = 3300  # секунд (чуть меньше часа жизни токена)

    def __init__(self, df, api_key: str, gigachat_token_url: str, gigachat_chat_url: str):
        self.df = df
        self.api_key = api_key
        self.TOKEN_URL = gigachat_token_url
        self.CHAT_URL = gigachat_chat_url
        self.tools = ToolRegistry(df)
        self.conversation_history: List[Dict] = []
        
        # Кэш токена на уровне экземпляра
        self._cached_token: Optional[str] = None
        self._token_expiry: float = 0.0
        
        # Системный промпт вынесен в константу для экономии ресурсов
        self._SYSTEM_PROMPT = self._build_system_prompt()

    @functools.lru_cache(maxsize=1)
    def _get_access_token(self) -> str:
        """Получение и кэширование токена. Обновляется автоматически при истечении TTL."""
        if self._cached_token and time.time() < self._token_expiry:
            return self._cached_token
            
        payload = "scope=GIGACHAT_API_PERS"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": f"Basic {self.api_key}"
        }
        
        response = requests.post(
            self.TOKEN_URL, data=payload, headers=headers, 
            verify=False, timeout=30
        )
        if response.status_code == 200:
            token_data = response.json()
            self._cached_token = token_data.get("access_token")
            self._token_expiry = time.time() + self.TOKEN_TTL
            return self._cached_token
            
        raise RuntimeError(f"Auth error {response.status_code}: {response.text}")

    def _parse_tool_call_from_text(self, text: str) -> Optional[Dict]:
        """Надёжный парсинг JSON-блока с вызовом инструмента из текста LLM"""
        # Ищем все потенциальные JSON-блоки
        brace_stack = []
        start_idx = None
        candidates = []
        
        for i, char in enumerate(text):
            if char == '{':
                if not brace_stack:
                    start_idx = i
                brace_stack.append('{')
            elif char == '}':
                if brace_stack:
                    brace_stack.pop()
                    if not brace_stack and start_idx is not None:
                        candidates.append(text[start_idx:i+1])
                        
        # Проверяем кандидатов с конца (последний часто самый точный)
        for candidate in reversed(candidates):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and "tool" in parsed and "arguments" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue
                
        return None

    def _build_system_prompt(self) -> str:
        """Генерация системного промпта один раз при инициализации"""
        tools_schema = json.dumps(self.tools.get_tool_schema(), ensure_ascii=False, indent=2)
        
        return f"""Ты — профессиональный аналитик данных с доступом к инструментам.

ДОСТУПНЫЕ ИНСТРУМЕНТЫ:
{tools_schema}

ВАЖНЫЕ ОГРАНИЧЕНИЯ:
- НЕ используй import в коде! Все модули уже импортированы: pandas (pd), numpy (np), math, statistics, json, re
- НЕ пытайся импортировать seaborn, matplotlib, scipy или другие библиотеки
- Используй ТОЛЬКО переменную 'df' для работы с данными
- Если инструмент вернул ошибку — ИСПРАВЬ код и повтори попытку, не генерируй текст об ошибке 

ПРАВИЛА РАБОТЫ:
1. Для анализа данных ВСЕГДА используй инструменты, а не догадки.
2. Если нужен расчёт — используй execute_python с pandas-โค้ด.
3. Формат вызова инструмента (строго в ответе):
   {{"tool": "имя_инструмента", "arguments": {{"параметр": "значение"}}}}
4. После получения результата инструмента — интерпретируй его для пользователя.
5. Если информации недостаточно — запроси дополнительные вычисления.
6. Отвечай на русском языке, структурированно, по делу.

ПРИМЕР:
User: Какая средняя продажа по категориям?
Assistant: {{"tool": "execute_python", "arguments": {{"code": "result = df.groupby('category')['sales'].mean().round(2)"}}}}
[Tool response: {{'success': True, 'result': {{'value': {{'A': 150.5, 'B': 230.1}}}}}}]
Assistant: Средняя продажа по категориям:
• Категория A: 150.5
• Категория B: 230.1
Вывод: Категория B лидирует...

НАЧНИ АНАЛИЗ."""

    def _call_llm(self, messages: List[Dict]) -> Tuple[str, Optional[Dict]]:
        """Вызов GigaChat API. Системный промпт добавляется автоматически."""
        access_token = self._get_access_token()
        
        # Формируем полный контекст запроса
        full_messages = [{"role": "system", "content": self._SYSTEM_PROMPT}] + messages
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}",
            "RqUID": str(uuid.uuid4())
        }
        
        body = {
            "model": "GigaChat",
            "messages": full_messages,
            "temperature": self.TEMPERATURE,
            "max_tokens": 3000
        }
        
        response = requests.post(
            self.CHAT_URL, json=body, headers=headers,
            verify=False, timeout=120
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"API error {response.status_code}: {response.text}")
        
        content = response.json()['choices'][0]['message']['content']
        tool_call = self._parse_tool_call_from_text(content)
        
        return content, tool_call

    def run(self, user_request: str, context: str = "") -> Dict[str, Any]:
        """
        Основной цикл агента.
        Returns: Dict с результатом анализа, логами и историей диалога.
        """
        start_time = time.time()
        logs = []
        
        # Инициализация диалога
        initial_message = f"Запрос пользователя: {user_request}"
        if context.strip():
            initial_message += f"\n\nКонтекст данных:\n{context.strip()}"
        
        self.conversation_history = [{"role": "user", "content": initial_message}]
        
        for iteration in range(self.MAX_ITERATIONS):
            logs.append(f"🔄 Итерация {iteration + 1}/{self.MAX_ITERATIONS}")
            
            # 1. Запрос к LLM
            llm_response, tool_call = self._call_llm(self.conversation_history)
            
            # 2. Сохраняем ответ ассистента в историю (важно для контекста!)
            self.conversation_history.append({"role": "assistant", "content": llm_response})
            
            # 3. Проверяем наличие вызова инструмента
            if tool_call:
                tool_name = tool_call['tool']
                arguments = tool_call['arguments']
                logs.append(f"🔧 Вызов инструмента: {tool_name}({arguments})")
                
                # 4. Выполнение инструмента
                tool_result = self.tools.execute(tool_name, **arguments)
                
                # 5. Формируем сообщение с результатом
                # GigaChat корректно обрабатывает role="tool", но для надёжности добавляем префикс
                tool_message_content = json.dumps(tool_result, ensure_ascii=False, default=str)
                self.conversation_history.append({
                    "role": "user", 
                    "content": f"[Результат {tool_name}]: {tool_message_content}"
                })
                logs.append(f"✅ Результат: {tool_result.get('success', False)}")
                
                if not tool_result.get('success', False):
                    logs.append(f"️ Ошибка инструмента: {tool_result.get('error')}")
                # Цикл продолжается автоматически на следующей итерации
            else:
                # 6. Если tool call нет — это финальный ответ
                logs.append("🎯 Финальный ответ получен")
                return {
                    'success': True,
                    'answer': llm_response,
                    'iterations': iteration + 1,
                    'execution_time': time.time() - start_time,
                    'logs': logs,
                    'conversation': self.conversation_history
                }
        
        # 7. Лимит итераций: принудительная финализация
        logs.append("⚠️ Достигнут лимит итераций. Формирую итоговый отчёт...")
        final_prompt = "На основе всей истории анализа сформулируй структурированный итоговый ответ пользователю на русском языке. Не запрашивай больше инструменты."
        self.conversation_history.append({"role": "user", "content": final_prompt})
        
        final_answer, _ = self._call_llm(self.conversation_history)
        
        return {
            'success': True,
            'answer': final_answer,
            'iterations': self.MAX_ITERATIONS,
            'execution_time': time.time() - start_time,
            'logs': logs,
            'conversation': self.conversation_history,
            'warning': 'Достигнут лимит итераций'
        }