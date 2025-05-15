import streamlit as st
import requests
import json
import os
import re
import time
import uuid
import sqlite3
from datetime import datetime
import tiktoken
from typing import List, Dict, Any, Optional, Generator
import hashlib
from functools import wraps, lru_cache
import traceback
import black

MODEL_OPTIONS = [
    {
        "id": "anthropic/claude-3.7-sonnet:floor",
        "name": "Claude 3.7 Sonnet",
        "pricing": {"prompt": 3.0, "completion": 15.0},
        "description": "Najnowszy model Claude z doskonałymi umiejętnościami kodowania"
    },
    {
        "id": "anthropic/claude-3.7-sonnet:thinking",
        "name": "Claude 3.7 Sonnet Thinking",
        "pricing": {"prompt": 3.0, "completion": 15.0},
        "description": "Model Claude wykorzystujący dodatkowy czas na analizę problemów"
    },
    {
        "id": "openai/gpt-4o:floor",
        "name": "GPT-4o",
        "pricing": {"prompt": 2.5, "completion": 10.0},
        "description": "Silna alternatywa z dobrymi zdolnościami kodowania"
    },
    {
        "id": "openai/gpt-4-turbo:floor",
        "name": "GPT-4 Turbo",
        "pricing": {"prompt": 2.5, "completion": 10.0},
        "description": "Nieco starszy model GPT-4 Turbo"
    },
    {
        "id": "anthropic/claude-3.5-haiku:floor",
        "name": "Claude 3.5 Haiku",
        "pricing": {"prompt": 0.8, "completion": 4.0},
        "description": "Szybszy, tańszy model do prostszych zadań"
    }
]

DEFAULT_SYSTEM_PROMPT = ""

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not st.session_state.get("authenticated", False):
            return login_page()
        return f(*args, **kwargs)
    return decorated

def login_page():
    st.title("🔒 Logowanie")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Zaloguj się, aby uzyskać dostęp")
        
        username = st.text_input("Nazwa użytkownika", key="login_username")
        password = st.text_input("Hasło", type="password", key="login_password")
        
        if st.button("Zaloguj"):
            correct_username = st.secrets.get("APP_USER", "admin")
            correct_password = st.secrets.get("APP_PASSWORD", "password")
            
            if username == correct_username and password == correct_password:
                st.session_state["authenticated"] = True
                st.success("Zalogowano pomyślnie!")
                st.rerun()
            else:
                st.error("Nieprawidłowa nazwa użytkownika lub hasło!")

class AssistantDB:
    def __init__(self, db_path='streamlit_assistant.db'):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
    
    def _create_tables(self):
        cursor = self.conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations (id)
        )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)')
        
        self.conn.commit()

    def save_message(self, conversation_id: str, role: str, content: str) -> str:
        cursor = self.conn.cursor()
        message_id = str(uuid.uuid4())
        
        cursor.execute(
            "INSERT INTO messages (id, conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
            (message_id, conversation_id, role, content, datetime.now())
        )
        self.conn.commit()
        return message_id
    
    def get_messages(self, conversation_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        
        query = "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY timestamp"
        params = [conversation_id]
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
            
        cursor.execute(query, params)
        
        messages = []
        for role, content in cursor.fetchall():
            messages.append({
                "role": role,
                "content": content
            })
                
        return messages

    def get_last_messages(self, conversation_id: str, count: int = 10) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY timestamp DESC LIMIT ?",
            (conversation_id, count)
        )
        
        messages = []
        for role, content in cursor.fetchall():
            messages.append({
                "role": role,
                "content": content
            })
                
        return messages[::-1]

    def get_message_count(self, conversation_id: str) -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
            (conversation_id,)
        )
        return cursor.fetchone()[0]

    def save_conversation(self, conversation_id: str, title: str):
        cursor = self.conn.cursor()
        now = datetime.now()
        
        cursor.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,))
        if cursor.fetchone():
            cursor.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, conversation_id)
            )
        else:
            cursor.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conversation_id, title, now, now)
            )
        
        self.conn.commit()
    
    def get_conversations(self) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, title, created_at FROM conversations ORDER BY updated_at DESC"
        )
        return [
            {"id": conv_id, "title": title, "created_at": created_at} 
            for conv_id, title, created_at in cursor.fetchall()
        ]

    def delete_conversation(self, conversation_id: str):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        self.conn.commit()

class LLMService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.cache = {}
        try:
            self.gpt4_encoding = tiktoken.encoding_for_model("gpt-4")
        except:
            self.gpt4_encoding = None
            
        try:
            self.claude_encoding = tiktoken.get_encoding("cl100k_base")
        except:
            self.claude_encoding = None

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
            
        try:
            if self.gpt4_encoding:
                return len(self.gpt4_encoding.encode(text))
            elif self.claude_encoding:
                return len(self.claude_encoding.encode(text))
            else:
                return len(text) // 4
        except:
            return len(text) // 4

    def get_cache_key(self, messages, model, system_prompt, temperature):
        cache_input = f"{json.dumps(messages)}-{model}-{system_prompt}-{temperature}"
        return hashlib.md5(cache_input.encode()).hexdigest()

    @lru_cache(maxsize=100)
    def get_cached_response(self, cache_key):
        return self.cache.get(cache_key)

    def call_llm(self, 
                messages: List[Dict[str, Any]], 
                model: str = "anthropic/claude-3.7-sonnet:floor", 
                system_prompt: str = None, 
                temperature: float = 0.7, 
                max_tokens: int = 12000,
                use_cache: bool = True) -> Dict[str, Any]:
        if use_cache and temperature < 0.1:  
            cache_key = self.get_cache_key(messages, model, system_prompt, temperature)
            cached_response = self.get_cached_response(cache_key)
            if cached_response:
                return cached_response
        
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        
        for msg in messages:
            api_messages.append({"role": msg["role"], "content": msg["content"]})
        
        prompt_text = system_prompt or ""
        for msg in messages:
            if isinstance(msg.get("content"), str):
                prompt_text += msg["content"]
        
        prompt_tokens = self.count_tokens(prompt_text)
        
        max_retries = 3
        retry_delay = 2  
        
        for attempt in range(max_retries):
            try:
                api_payload = {
                    "model": model,
                    "messages": api_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=api_payload,
                    timeout=300 
                )
                response.raise_for_status()
                result = response.json()
                
                response_content = result["choices"][0]["message"]["content"]
                completion_tokens = self.count_tokens(response_content)
                
                result["usage"] = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
                
                if use_cache and temperature < 0.1:
                    cache_key = self.get_cache_key(messages, model, system_prompt, temperature)
                    self.cache[cache_key] = result
                
                return result
            
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:  
                    raise Exception(f"Nie udało się połączyć z API po {max_retries} próbach: {str(e)}")
                
                time.sleep(retry_delay * (2 ** attempt))  
    
    def call_llm_streaming(self, 
                         messages: List[Dict[str, Any]], 
                         model: str = "anthropic/claude-3.7-sonnet:floor", 
                         system_prompt: str = None, 
                         temperature: float = 0.7, 
                         max_tokens: int = 12000,
                         optimize_code: bool = True) -> Generator[str, None, None]:
        
        if optimize_code and system_prompt:
            system_prompt += "\nJeśli podajesz długie fragmenty kodu (ponad 500 linii), staraj się minimalizować zbędne spacje i komentarze. Kod zostanie sformatowany po stronie klienta."
        elif optimize_code:
            system_prompt = "Jeśli podajesz długie fragmenty kodu (ponad 500 linii), staraj się minimalizować zbędne spacje i komentarze. Kod zostanie sformatowany po stronie klienta."
        
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        
        for msg in messages:
            api_messages.append({"role": msg["role"], "content": msg["content"]})
        
        prompt_text = system_prompt or ""
        for msg in messages:
            if isinstance(msg.get("content"), str):
                prompt_text += msg["content"]
        
        prompt_tokens = self.count_tokens(prompt_text)
        
        max_retries = 3
        retry_delay = 2  
        last_error = None
        
        code_start_pattern = re.compile(r"```([a-zA-Z0-9]*)")
        code_end_pattern = re.compile(r"```")
        
        for attempt in range(max_retries):
            try:
                api_payload = {
                    "model": model,
                    "messages": api_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": True
                }
                
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=api_payload,
                    stream=True,
                    timeout=600
                )
                
                response.raise_for_status()
                
                full_response = ""
                completion_tokens = 0
                
                in_code_block = False
                current_code = ""
                current_lang = ""
                code_blocks = {}
                current_block_id = 0
                
                buffer = ""
                chunk_collection = []
                last_update_time = time.time()
                
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data:'):
                            if line.strip() == 'data: [DONE]':
                                break
                            
                            try:
                                data = json.loads(line[5:])
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        content_chunk = delta["content"]
                                        chunk_collection.append(content_chunk)
                                        
                                        buffer += content_chunk
                                        
                                        if len(chunk_collection) % 100 == 0:
                                            if "```" in buffer and not in_code_block:
                                                in_code_block = True
                                                block_start = buffer.rindex("```")
                                                before_code = buffer[:block_start]
                                                code_start = buffer[block_start+3:]
                                                
                                                if "\n" in code_start:
                                                    first_line, rest = code_start.split("\n", 1)
                                                    current_lang = first_line.strip()
                                                    current_code = rest
                                                else:
                                                    current_lang = ""
                                                    current_code = code_start
                                                
                                                if before_code:
                                                    full_response += before_code
                                                
                                                current_block_id += 1
                                                code_blocks[current_block_id] = {"code": current_code, "lang": current_lang}
                                                full_response += f"```{current_lang}\n[CODE_BLOCK_{current_block_id}]\n```"
                                                buffer = ""
                                            
                                            elif "```" in buffer and in_code_block:
                                                in_code_block = False
                                                code_end = buffer.index("```")
                                                more_code = buffer[:code_end]
                                                after_code = buffer[code_end+3:]
                                                
                                                code_blocks[current_block_id]["code"] += more_code
                                                
                                                if after_code:
                                                    full_response += after_code
                                                
                                                buffer = ""
                                            
                                            elif in_code_block:
                                                code_blocks[current_block_id]["code"] += buffer
                                                buffer = ""
                                            
                                            elif not in_code_block:
                                                full_response += buffer
                                                buffer = ""
                                        
                                        current_time = time.time()
                                        if len(full_response) % 1000 == 0 or (current_time - last_update_time) > 2.0:
                                            st.session_state["current_stream_response"] = full_response
                                            st.session_state["code_blocks"] = code_blocks
                                            last_update_time = current_time
                                        
                                        completion_tokens += self.count_tokens(content_chunk)
                                        
                                        yield content_chunk
                            except json.JSONDecodeError:
                                pass
                
                if buffer:
                    if in_code_block:
                        code_blocks[current_block_id]["code"] += buffer
                    else:
                        full_response += buffer
                
                st.session_state["current_stream_response"] = full_response
                st.session_state["code_blocks"] = code_blocks
                
                yield {
                    "full_response": full_response,
                    "code_blocks": code_blocks,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                }
                
                break
                
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                if attempt < max_retries - 1:  
                    time.sleep(retry_delay * (2 ** attempt))  
                    continue
                
                error_info = {
                    "error": True,
                    "error_message": f"Wystąpił błąd podczas streamowania: {last_error}",
                    "partial_response": st.session_state.get("current_stream_response", ""),
                    "code_blocks": st.session_state.get("code_blocks", {})
                }
                yield error_info

def prepare_messages_with_token_management(messages, system_prompt, model_id, llm_service):
    model_token_limits = {
        "anthropic/claude-3.7-sonnet:floor": 180000,
        "anthropic/claude-3.7-sonnet:thinking": 180000,
        "anthropic/claude-3.5-haiku:floor": 150000,
        "openai/gpt-4o:floor": 120000, 
        "openai/gpt-4-turbo:floor": 100000,
    }
    
    default_token_limit = 100000
    max_input_tokens = model_token_limits.get(model_id, default_token_limit)
    
    max_completion_tokens = 12000
    system_tokens = llm_service.count_tokens(system_prompt) if system_prompt else 0
    available_tokens = max_input_tokens - system_tokens - max_completion_tokens - 100  
    
    if not messages:
        return []

    api_messages = []
    if system_prompt:
        api_messages.append({"role": "system", "content": system_prompt})

    last_user_msg = None
    last_user_idx = -1
    for i in range(len(messages)-1, -1, -1):
        if messages[i]["role"] == "user":
            last_user_msg = messages[i]
            last_user_idx = i
            break

    last_assistant_msg = None
    if last_user_idx > 0:
        for i in range(last_user_idx-1, -1, -1):
            if messages[i]["role"] == "assistant":
                last_assistant_msg = messages[i]
                break

    first_msg = messages[0] if messages else None

    last_user_tokens = llm_service.count_tokens(last_user_msg["content"]) if last_user_msg else 0
    last_assistant_tokens = llm_service.count_tokens(last_assistant_msg["content"]) if last_assistant_msg else 0
    first_msg_tokens = llm_service.count_tokens(first_msg["content"]) if first_msg else 0
    
    important_tokens = last_user_tokens + last_assistant_tokens + first_msg_tokens
    
    if important_tokens <= available_tokens:
        if first_msg and first_msg != last_user_msg and first_msg != last_assistant_msg:
            api_messages.append(first_msg)
            available_tokens -= first_msg_tokens
        
        if last_assistant_msg:
            api_messages.append(last_assistant_msg)
            available_tokens -= last_assistant_tokens
        
        if last_user_msg:
            api_messages.append(last_user_msg)
            available_tokens -= last_user_tokens
        
        current_tokens = system_tokens + important_tokens
        
        remaining_messages = [msg for idx, msg in enumerate(messages) 
                              if msg != first_msg and msg != last_assistant_msg and msg != last_user_msg]
        
        for msg in remaining_messages:
            msg_tokens = llm_service.count_tokens(msg["content"])
            
            if current_tokens + msg_tokens <= max_input_tokens - max_completion_tokens - 100:
                api_messages.append(msg)
                current_tokens += msg_tokens
            else:
                break
        
        api_messages.sort(key=lambda x: messages.index(x) if x in messages else -1)
    else:
        if last_user_msg:
            if last_user_tokens <= available_tokens:
                api_messages.append(last_user_msg)
            else:
                truncated_content = last_user_msg["content"][:int(available_tokens * 4)]
                api_messages.append({"role": "user", "content": truncated_content})
                
    return api_messages

def process_code_blocks(content, optimize_code=True):
    if "```" not in content:
        return content

    code_block_pattern = re.compile(r"```([a-zA-Z0-9]*)\n(.*?)\n```", re.DOTALL)

    def replace_block(match):
        lang = match.group(1) or "python"
        code = match.group(2)

        if lang == "python" and optimize_code and len(code.splitlines()) > 10:
            try:
                code = format_code_with_black(code)
            except:
                pass

        return f"```{lang}\n{code}\n```"

    return code_block_pattern.sub(replace_block, content)

def format_message_for_display(message):
    content = message.get("content", "")
    return process_code_blocks(content, optimize_code=True)

def get_conversation_title(messages, llm_service, api_key):
    if not messages:
        return f"Nowa konwersacja {datetime.now().strftime('%d-%m-%Y %H:%M')}"
    
    user_message = next((m["content"] for m in messages if m["role"] == "user"), "")
    
    if len(user_message) > 40:
        try:
            response = llm_service.call_llm(
                messages=[
                    {"role": "user", "content": f"Utwórz krótki, opisowy tytuł (max. 5 słów) dla następującej konwersacji, bez cudzysłowów: {user_message[:200]}..."}
                ],
                model="anthropic/claude-3.5-haiku:floor",  
                system_prompt="Jesteś pomocnym asystentem, który tworzy krótkie, opisowe tytuły konwersacji.",
                temperature=0.2,
                max_tokens=20,
                use_cache=True
            )
            title = response["choices"][0]["message"]["content"].strip().strip('"\'')
            title = re.sub(r'[^\w\s\-.,]', '', title)
            return title[:40]
        except Exception:
            pass
    
    return user_message[:40] + ("..." if len(user_message) > 40 else "")

def parse_code_blocks(content):
    if not content or "```" not in content:
        return []
        
    code_blocks = []
    pattern = re.compile(r"```([a-zA-Z0-9]*)\n(.*?)\n```", re.DOTALL)
    matches = pattern.findall(content)
    
    for lang, code in matches:
        language = lang.strip() if lang.strip() else "python"
        code_blocks.append({"language": language, "code": code})
    
    return code_blocks

def format_code_with_black(code, line_length=88):
    try:
        return black.format_str(code, mode=black.Mode(line_length=line_length))
    except:
        return code

def sidebar_component():
    st.sidebar.title("AI Asystent Developera")
    
    with st.sidebar.expander("⚙️ Ustawienia modelu", expanded=False):
        model_options = {model["id"]: f"{model['name']}" for model in MODEL_OPTIONS}
        selected_model = st.selectbox(
            "Model LLM",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0,
            key="model_selection"
        )
        
        for model in MODEL_OPTIONS:
            if model["id"] == selected_model:
                st.info(model["description"])
        
        temperature = st.slider(
            "Temperatura",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("temperature", 0.7),
            step=0.1,
            help="Wyższa wartość = bardziej kreatywne odpowiedzi"
        )
        
        st.session_state["temperature"] = temperature
        
        custom_system_prompt = st.text_area(
            "Prompt systemowy (opcjonalnie)",
            value=st.session_state.get("custom_system_prompt", DEFAULT_SYSTEM_PROMPT),
            help="Dostosuj zachowanie asystenta"
        )
        
        if st.button("Zresetuj do domyślnego"):
            custom_system_prompt = DEFAULT_SYSTEM_PROMPT
        
        st.session_state["custom_system_prompt"] = custom_system_prompt
        
        optimize_code = st.checkbox(
            "Optymalizuj długie kody",
            value=st.session_state.get("optimize_code", True),
            help="Formatuje otrzymane bloki kodu za pomocą black"
        )
        
        st.session_state["optimize_code"] = optimize_code
    
    if "token_usage" in st.session_state:
        with st.sidebar.expander("📊 Statystyki tokenów", expanded=False):
            st.metric("Tokeny prompt", st.session_state["token_usage"]["prompt"])
            st.metric("Tokeny completion", st.session_state["token_usage"]["completion"])
            st.metric("Szacunkowy koszt", f"${st.session_state['token_usage']['cost']:.4f}")
    
    db = st.session_state.get("db")
    if db:
        with st.sidebar.expander("💬 Konwersacje", expanded=True):
            conversations = db.get_conversations()
            
            if conversations:
                for conv in conversations:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        if st.button(conv["title"], key=f"conv_{conv['id']}", use_container_width=True):
                            st.session_state["current_conversation_id"] = conv["id"]
                            st.rerun()
                    with col2:
                        if st.button("🗑️", key=f"del_{conv['id']}", help="Usuń konwersację"):
                            db.delete_conversation(conv["id"])
                            if st.session_state.get("current_conversation_id") == conv["id"]:
                                st.session_state["current_conversation_id"] = None
                            st.rerun()
            else:
                st.write("Brak zapisanych konwersacji")
        
        if st.sidebar.button("➕ Nowa konwersacja", use_container_width=True):
            st.session_state["current_conversation_id"] = str(uuid.uuid4())
            if "current_stream_response" in st.session_state:
                del st.session_state["current_stream_response"]
            st.rerun()

def calculate_cost(model_id, prompt_tokens, completion_tokens):
    for model in MODEL_OPTIONS:
        if model["id"] == model_id:
            return (prompt_tokens / 1_000_000) * model["pricing"]["prompt"] + \
                   (completion_tokens / 1_000_000) * model["pricing"]["completion"]
    return 0.0

@requires_auth
def chat_component():
    st.markdown("""
    <style>
    .main .block-container {
        padding-bottom: 80px;
    }
    
    .stChatInputContainer {
        position: fixed;
        bottom: 0;
        left: 240px; 
        right: 0;
        padding: 1rem;
        background: white;
        z-index: 999;
        border-top: 1px solid #ddd;
    }
    
    .stMarkdown pre {
        overflow-x: auto;
    }
    
    .element-container .stMarkdown {
        min-height: 20px;
    }
    
    @media (max-width: 768px) {
        .stChatInputContainer {
            left: 0;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    api_key = st.secrets.get("OPENROUTER_API_KEY", "")
    if not api_key:
        st.warning("⚠️ Brak klucza API OpenRouter w ustawieniach secrets.")
        return

    st.session_state["api_key"] = api_key

    if "llm_service" not in st.session_state or st.session_state.get("llm_service") is None:
        st.session_state["llm_service"] = LLMService(api_key)

    llm_service = st.session_state.get("llm_service")
    db = st.session_state.get("db")

    if not llm_service or not db:
        st.error("⚠️ Nie można zainicjalizować serwisów. Odśwież stronę i spróbuj ponownie.")
        return

    if "current_conversation_id" not in st.session_state:
        st.session_state["current_conversation_id"] = str(uuid.uuid4())

    current_conversation_id = st.session_state["current_conversation_id"]

    if "token_usage" not in st.session_state:
        st.session_state["token_usage"] = {"prompt": 0, "completion": 0, "cost": 0.0}
    
    if "current_stream_response" not in st.session_state:
        st.session_state["current_stream_response"] = ""
        
    if "code_blocks" not in st.session_state:
        st.session_state["code_blocks"] = {}

    messages = db.get_messages(current_conversation_id, limit=50)
    
    message_containers = st.container()
    
    with message_containers:
        for message in messages:
            role = message["role"]
            content = format_message_for_display(message)

            if role == "user":
                with st.chat_message("user"):
                    st.markdown(content)
            elif role == "assistant":
                with st.chat_message("assistant"):
                    processed_content = content
                    code_block_pattern = r"```([a-zA-Z0-9]*)\n\[CODE_BLOCK_(\d+)\]\n```"
                    code_blocks_in_message = {}
                    
                    regular_code_blocks = parse_code_blocks(content)
                    for i, block in enumerate(regular_code_blocks):
                        lang = block["language"]
                        code = block["code"]
                        
                        if lang == "python" and st.session_state.get("optimize_code", True) and len(code.splitlines()) > 10:
                            try:
                                code = format_code_with_black(code)
                            except:
                                pass
                        
                        code_blocks_in_message[i+1] = {"code": code, "lang": lang}
                    
                    compiled_pattern = re.compile(code_block_pattern)
                    for match in compiled_pattern.finditer(processed_content):
                        lang, block_id = match.groups()
                        block_id = int(block_id)
                        code_marker = match.group(0)
                        
                        if block_id in code_blocks_in_message:
                            code = code_blocks_in_message[block_id]["code"]
                            processed_content = processed_content.replace(code_marker, f"```{lang}\n{code}\n```")
                    
                    st.markdown(processed_content)
    
    if "current_stream_response" in st.session_state and st.session_state["current_stream_response"]:
        if st.session_state.get("stream_was_interrupted", False):
            with st.chat_message("assistant"):
                st.warning("⚠️ Poprzednia odpowiedź została przerwana. Oto częściowa odpowiedź:")
                
                partial_response = st.session_state["current_stream_response"]
                code_blocks = st.session_state.get("code_blocks", {})
                
                for block_id, block_data in code_blocks.items():
                    marker = f"```{block_data['lang']}\n[CODE_BLOCK_{block_id}]\n```"
                    code = block_data["code"]
                    
                    if block_data['lang'] == "python" and st.session_state.get("optimize_code", True) and len(code.splitlines()) > 10:
                        try:
                            code = format_code_with_black(code)
                        except:
                            pass
                    
                    partial_response = partial_response.replace(marker, f"```{block_data['lang']}\n{code}\n```")
                
                st.markdown(partial_response)
                
                save_partial_container = st.empty()
                
                if save_partial_container.button("Zapisz tę częściową odpowiedź"):
                    db.save_message(current_conversation_id, "assistant", partial_response)
                    st.session_state["current_stream_response"] = ""
                    st.session_state["code_blocks"] = {}
                    st.session_state["stream_was_interrupted"] = False
                    save_partial_container.success("Zapisano częściową odpowiedź!")
                    time.sleep(1)
                    st.rerun()
    
    user_input = st.chat_input("Wpisz swoje pytanie lub zadanie...")

    if user_input:
        st.session_state["stream_was_interrupted"] = False
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        if len(messages) == 0:
            conversation_title = get_conversation_title([{"role": "user", "content": user_input}], llm_service, st.session_state["api_key"])
            db.save_conversation(current_conversation_id, conversation_title)
        
        db.save_message(current_conversation_id, "user", user_input)
        
        if len(messages) > 100:
            first_message = db.get_messages(current_conversation_id, limit=1)
            recent_messages = db.get_last_messages(current_conversation_id, count=50)
            all_messages = first_message + recent_messages
        else:
            all_messages = db.get_messages(current_conversation_id)
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            full_response = ""
            code_blocks = {}
            
            try:
                model = st.session_state.get("model_selection", MODEL_OPTIONS[0]["id"])
                system_prompt = st.session_state.get("custom_system_prompt", DEFAULT_SYSTEM_PROMPT)
                temperature = st.session_state.get("temperature", 0.7)
                optimize_code = st.session_state.get("optimize_code", True)
                
                optimized_messages = prepare_messages_with_token_management(
                    all_messages, 
                    system_prompt, 
                    model, 
                    llm_service
                )
                
                st.session_state["current_stream_response"] = ""
                st.session_state["code_blocks"] = {}
                st.session_state["stream_was_interrupted"] = False
                
                metadata = None
                
                try:
                    for chunk in llm_service.call_llm_streaming(
                        messages=optimized_messages,
                        model=model,
                        system_prompt=None,
                        temperature=temperature,
                        max_tokens=12000,
                        optimize_code=optimize_code
                    ):
                        if isinstance(chunk, dict) and "error" in chunk and chunk["error"]:
                            st.error(chunk["error_message"])
                            full_response = chunk.get("partial_response", "")
                            code_blocks = chunk.get("code_blocks", {})
                            st.session_state["current_stream_response"] = full_response
                            st.session_state["code_blocks"] = code_blocks
                            st.session_state["stream_was_interrupted"] = True
                            
                            processed_response = process_partial_response(full_response, code_blocks, optimize_code)
                            
                            response_placeholder.markdown(processed_response)
                            break
                        
                        if isinstance(chunk, dict) and "full_response" in chunk:
                            metadata = chunk
                            full_response = metadata["full_response"] 
                            code_blocks = metadata.get("code_blocks", {})
                            break
                        
                        if not isinstance(chunk, dict):
                            display_response = process_streaming_update()
                            
                            response_placeholder.markdown(display_response + "▌")
                        
                except Exception as e:
                    st.error(f"Błąd podczas streamowania: {str(e)}")
                    st.session_state["stream_was_interrupted"] = True
                    response_placeholder.markdown(full_response)
                    return
                
                if metadata:
                    full_response = metadata["full_response"]
                    code_blocks = metadata.get("code_blocks", {})
                
                processed_response = process_partial_response(full_response, code_blocks, optimize_code)
                
                response_placeholder.markdown(processed_response)
                
                if metadata and "usage" in metadata:
                    usage = metadata["usage"]
                    st.session_state["token_usage"]["prompt"] += usage["prompt_tokens"]
                    st.session_state["token_usage"]["completion"] += usage["completion_tokens"]
                    
                    cost = calculate_cost(
                        model, 
                        usage["prompt_tokens"], 
                        usage["completion_tokens"]
                    )
                    
                    st.session_state["token_usage"]["cost"] += cost
                
                db.save_message(current_conversation_id, "assistant", processed_response)
                
                st.session_state["current_stream_response"] = ""
                st.session_state["code_blocks"] = {}
                
                st.rerun()
                
            except Exception as e:
                st.session_state["stream_was_interrupted"] = True
                
                st.error(f"Wystąpił błąd: {str(e)}")
                st.code(traceback.format_exc())
                
                if full_response:
                    st.warning("Częściowa odpowiedź przed wystąpieniem błędu:")
                    
                    processed_response = process_partial_response(full_response, code_blocks, optimize_code)
                    
                    st.markdown(processed_response)
                    st.session_state["current_stream_response"] = full_response
                    
                    save_button = st.empty()
                    if save_button.button("Zapisz tę częściową odpowiedź"):
                        db.save_message(current_conversation_id, "assistant", processed_response)
                        st.session_state["current_stream_response"] = ""
                        st.session_state["code_blocks"] = {}
                        save_button.success("Zapisano częściową odpowiedź!")
                        time.sleep(1)
                        st.rerun()

def process_streaming_update():
    full_response = st.session_state.get("current_stream_response", "")
    code_blocks = st.session_state.get("code_blocks", {})
    
    display_response = full_response
    for block_id, block_data in code_blocks.items():
        marker = f"```{block_data['lang']}\n[CODE_BLOCK_{block_id}]\n```"
        if marker in display_response:
            display_response = display_response.replace(marker, f"```{block_data['lang']}\n{block_data['code']}\n```")
    
    return display_response

def process_partial_response(response, code_blocks, optimize_code):
    processed_response = response
    
    if "[CODE_BLOCK_" not in processed_response:
        return processed_response
        
    for block_id, block_data in code_blocks.items():
        marker = f"```{block_data['lang']}\n[CODE_BLOCK_{block_id}]\n```"
        code = block_data["code"]
        
        if block_data['lang'] == "python" and optimize_code and len(code.splitlines()) > 10:
            try:
                code = format_code_with_black(code)
            except:
                pass
        
        processed_response = processed_response.replace(marker, f"```{block_data['lang']}\n{code}\n```")
    
    return processed_response

def main():
    st.set_page_config(
        page_title="AI Asystent Developera Streamlit",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    if "db" not in st.session_state:
        st.session_state["db"] = AssistantDB()
    
    if not st.session_state.get("authenticated", False):
        login_page()
        return
    
    api_key = st.secrets.get("OPENROUTER_API_KEY", "")
    if api_key and "llm_service" not in st.session_state:
        st.session_state["llm_service"] = LLMService(api_key)
        st.session_state["api_key"] = api_key
    
    sidebar_component()
    
    chat_component()

if __name__ == "__main__":
    main()
