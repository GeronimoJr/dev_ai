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
from functools import wraps
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
    
    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY timestamp",
            (conversation_id,)
        )
        
        messages = []
        for role, content in cursor.fetchall():
            messages.append({
                "role": role,
                "content": content
            })
                
        return messages

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
    
    api_messages = []
    if system_prompt:
        api_messages.append({"role": "system", "content": system_prompt})
    
    current_tokens = 0
    user_messages = []
    assistant_messages = []
    
    for msg in messages:
        if msg["role"] == "user":
            user_messages.append(msg)
        else:
            assistant_messages.append(msg)
    
    if user_messages:
        last_user_message = user_messages.pop()
    else:
        last_user_message = None
    
    if assistant_messages:
        last_assistant_message = assistant_messages.pop()
    else:
        last_assistant_message = None
    
    if user_messages:
        first_user_message = user_messages.pop(0)
    else:
        first_user_message = None
    
    if first_user_message:
        first_msg_tokens = llm_service.count_tokens(first_user_message["content"])
        if current_tokens + first_msg_tokens <= available_tokens:
            api_messages.append(first_user_message)
            current_tokens += first_msg_tokens
    
    remaining_messages = []
    i, j = 0, 0
    while i < len(user_messages) or j < len(assistant_messages):
        if i < len(user_messages):
            remaining_messages.append(user_messages[i])
            i += 1
        if j < len(assistant_messages):
            remaining_messages.append(assistant_messages[j])
            j += 1
    
    for msg in remaining_messages:
        msg_tokens = llm_service.count_tokens(msg["content"])
        
        if current_tokens + msg_tokens > available_tokens:
            break
        
        api_messages.append(msg)
        current_tokens += msg_tokens
    
    if last_assistant_message:
        last_assistant_tokens = llm_service.count_tokens(last_assistant_message["content"])
        if current_tokens + last_assistant_tokens <= available_tokens:
            api_messages.append(last_assistant_message)
            current_tokens += last_assistant_tokens
        else:
            truncated_content = "POPRZEDNIA ODPOWIEDŹ (skrócona): " + last_assistant_message["content"][:1000] + "..."
            truncated_tokens = llm_service.count_tokens(truncated_content)
            if current_tokens + truncated_tokens <= available_tokens:
                api_messages.append({"role": "assistant", "content": truncated_content})
                current_tokens += truncated_tokens
    
    if last_user_message:
        last_user_tokens = llm_service.count_tokens(last_user_message["content"])
        
        if current_tokens + last_user_tokens > available_tokens:
            remaining_tokens = available_tokens - current_tokens
            truncated_content = last_user_message["content"][:remaining_tokens * 4]  
            modified_message = {"role": "user", "content": truncated_content}
            api_messages.append(modified_message)
        else:
            api_messages.append(last_user_message)
    
    return api_messages

def format_message_for_display(message: Dict[str, str]) -> str:
    content = message.get("content", "")
    
    def replace_code_block(match):
        lang = match.group(1) or ""
        code = match.group(2)
        return f"```{lang}\n{code}\n```"
    
    content = re.sub(r"```(.*?)\n(.*?)```", replace_code_block, content, flags=re.DOTALL)
    
    return content

def get_conversation_title(messages: List[Dict[str, str]], llm_service: LLMService, api_key: str) -> str:
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
    if not content:
        return []
        
    code_blocks = []
    pattern = r"```([a-zA-Z0-9]*)\n(.*?)\n```"
    matches = re.findall(pattern, content, re.DOTALL)
    
    for lang, code in matches:
        language = lang.strip() if lang.strip() else "python"
        code_blocks.append({"language": language, "code": code})
    
    return code_blocks

def format_code_with_black(code, line_length=88):
    """Formatuje kod Python za pomocą black"""
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

class LLMService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.cache = {}

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
            
        try:
            encoding = tiktoken.encoding_for_model("gpt-4")  
            return len(encoding.encode(text))
        except:
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))
            except Exception as e:
                return len(text) // 4  

    def get_cache_key(self, messages, model, system_prompt, temperature):
        cache_input = f"{json.dumps(messages)}-{model}-{system_prompt}-{temperature}"
        return hashlib.md5(cache_input.encode()).hexdigest()

    def call_llm(self, 
                messages: List[Dict[str, Any]], 
                model: str = "anthropic/claude-3.7-sonnet:floor", 
                system_prompt: str = None, 
                temperature: float = 0.7, 
                max_tokens: int = 12000,
                use_cache: bool = True) -> Dict[str, Any]:
        if use_cache and temperature < 0.1:  
            cache_key = self.get_cache_key(messages, model, system_prompt, temperature)
            if cache_key in self.cache:
                return self.cache[cache_key]
        
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
        
        # Dodanie instrukcji dotyczącej zwięzłego formatowania kodu
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
                    timeout=600  # Zwiększamy timeout dla długich odpowiedzi
                )
                
                response.raise_for_status()
                
                full_response = ""
                completion_tokens = 0
                
                # Zmienne do śledzenia bloków kodu
                in_code_block = False
                current_code = ""
                current_lang = ""
                code_blocks = {}  # Słownik bloków kodu {id: {"code": code, "lang": lang}}
                current_block_id = 0
                
                # Bufor do zbierania fragmentów
                buffer = ""
                chunk_collection = []
                
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
                                        
                                        # Akumulujemy fragmenty w buforze
                                        buffer += content_chunk
                                        
                                        # Co 50 fragmentów sprawdzamy, czy mamy kompletne bloki kodu
                                        if len(chunk_collection) % 50 == 0:
                                            # Wykrywanie bloków kodu (start)
                                            if "```" in buffer and not in_code_block:
                                                in_code_block = True
                                                block_start = buffer.rindex("```")
                                                before_code = buffer[:block_start]
                                                code_start = buffer[block_start+3:]
                                                
                                                if "\n" in code_start:
                                                    # Wykrywanie języka
                                                    first_line, rest = code_start.split("\n", 1)
                                                    current_lang = first_line.strip()
                                                    current_code = rest
                                                else:
                                                    current_lang = ""
                                                    current_code = code_start
                                                
                                                # Dodaj tekst przed kodem
                                                if before_code:
                                                    full_response += before_code
                                                    st.session_state["current_stream_response"] = full_response
                                                
                                                # Przygotuj marker dla bloku kodu
                                                current_block_id += 1
                                                code_blocks[current_block_id] = {"code": current_code, "lang": current_lang}
                                                full_response += f"```{current_lang}\n[CODE_BLOCK_{current_block_id}]\n```"
                                                buffer = ""
                                            
                                            # Wykrywanie bloków kodu (koniec)
                                            elif "```" in buffer and in_code_block:
                                                in_code_block = False
                                                code_end = buffer.index("```")
                                                more_code = buffer[:code_end]
                                                after_code = buffer[code_end+3:]
                                                
                                                # Zaktualizuj aktualny blok kodu
                                                code_blocks[current_block_id]["code"] += more_code
                                                
                                                # Dodaj tekst po kodzie
                                                if after_code:
                                                    full_response += after_code
                                                
                                                buffer = ""
                                            
                                            # Aktualizacja kodu w trakcie bloku
                                            elif in_code_block:
                                                code_blocks[current_block_id]["code"] += buffer
                                                buffer = ""
                                            
                                            # Zwykły tekst poza blokiem kodu
                                            elif not in_code_block:
                                                full_response += buffer
                                                buffer = ""
                                        
                                        # Co 500 znaków zapisujemy postęp
                                        if len(full_response) % 500 == 0:
                                            st.session_state["current_stream_response"] = full_response
                                            # Aktualizujemy też słownik bloków kodu
                                            st.session_state["code_blocks"] = code_blocks
                                        
                                        completion_tokens += self.count_tokens(content_chunk)
                                        
                                        # Wysyłamy fragmenty jako generator
                                        yield content_chunk
                            except json.JSONDecodeError:
                                pass
                
                # Obsługa pozostałego bufora
                if buffer:
                    if in_code_block:
                        code_blocks[current_block_id]["code"] += buffer
                    else:
                        full_response += buffer
                
                # Zapisujemy pełną odpowiedź i bloki kodu
                st.session_state["current_stream_response"] = full_response
                st.session_state["code_blocks"] = code_blocks
                
                # Zwracamy metadane jako ostatni element
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

def calculate_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
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

    messages = db.get_messages(current_conversation_id)
    
    # Wyświetl istniejące wiadomości
    for message in messages:
        role = message["role"]
        content = format_message_for_display(message)

        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        elif role == "assistant":
            with st.chat_message("assistant"):
                # Sprawdzanie i przetwarzanie specjalnych markerów bloków kodu
                processed_content = content
                code_block_pattern = r"```([a-zA-Z0-9]*)\n\[CODE_BLOCK_(\d+)\]\n```"
                code_blocks_in_message = {}
                
                # Wyszukaj standardowe bloki kodu
                regular_code_blocks = parse_code_blocks(content)
                for i, block in enumerate(regular_code_blocks):
                    lang = block["language"]
                    code = block["code"]
                    
                    # Formatuj kod Python za pomocą black, jeśli włączona optymalizacja
                    if lang == "python" and st.session_state.get("optimize_code", True) and len(code.splitlines()) > 10:
                        try:
                            code = format_code_with_black(code)
                        except:
                            pass
                    
                    # Przechowaj kod do późniejszego wyświetlenia
                    code_blocks_in_message[i+1] = {"code": code, "lang": lang}
                
                # Jeśli znaleziono specjalne markery bloków kodu
                for match in re.finditer(code_block_pattern, processed_content):
                    lang, block_id = match.groups()
                    block_id = int(block_id)
                    code_marker = match.group(0)
                    
                    # Jeśli mamy kod dla tego markera, zastąp go rzeczywistym kodem
                    if block_id in code_blocks_in_message:
                        code = code_blocks_in_message[block_id]["code"]
                        processed_content = processed_content.replace(code_marker, f"```{lang}\n{code}\n```")
                
                # Wyświetl przetworzony tekst
                st.markdown(processed_content)
    
    # Sprawdź, czy istnieje częściowa odpowiedź, którą trzeba wyświetlić (po przerwaniu)
    if "current_stream_response" in st.session_state and st.session_state["current_stream_response"]:
        if st.session_state.get("stream_was_interrupted", False):
            with st.chat_message("assistant"):
                st.warning("⚠️ Poprzednia odpowiedź została przerwana. Oto częściowa odpowiedź:")
                
                # Przetwarzamy zawartość, aby zastąpić markery bloków kodu rzeczywistym kodem
                partial_response = st.session_state["current_stream_response"]
                code_blocks = st.session_state.get("code_blocks", {})
                
                for block_id, block_data in code_blocks.items():
                    marker = f"```{block_data['lang']}\n[CODE_BLOCK_{block_id}]\n```"
                    code = block_data["code"]
                    
                    # Formatuj kod Python za pomocą black, jeśli włączona optymalizacja
                    if block_data['lang'] == "python" and st.session_state.get("optimize_code", True) and len(code.splitlines()) > 10:
                        try:
                            code = format_code_with_black(code)
                        except:
                            pass
                    
                    partial_response = partial_response.replace(marker, f"```{block_data['lang']}\n{code}\n```")
                
                st.markdown(partial_response)
                
                # Opcja zapisania częściowej odpowiedzi
                if st.button("Zapisz tę częściową odpowiedź"):
                    db.save_message(current_conversation_id, "assistant", partial_response)
                    st.session_state["current_stream_response"] = ""
                    st.session_state["code_blocks"] = {}
                    st.session_state["stream_was_interrupted"] = False
                    st.success("Zapisano częściową odpowiedź!")
                    st.rerun()
    
    # Pole wejściowe użytkownika
    user_input = st.chat_input("Wpisz swoje pytanie lub zadanie...")

    # Obsługa wprowadzonego komunikatu
    if user_input:
        st.session_state["stream_was_interrupted"] = False
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Sprawdź, czy konwersacja ma tytuł
        if len(messages) == 0:
            conversation_title = get_conversation_title([{"role": "user", "content": user_input}], llm_service, st.session_state["api_key"])
            db.save_conversation(current_conversation_id, conversation_title)
        
        db.save_message(current_conversation_id, "user", user_input)
        
        # Pobierz wszystkie wiadomości, aby zachować kontekst
        all_messages = db.get_messages(current_conversation_id)
        
        # Wyświetl oczekującą odpowiedź asystenta
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            # Inicjalizuj zmienne
            full_response = ""
            code_blocks = {}
            
            try:
                model = st.session_state.get("model_selection", MODEL_OPTIONS[0]["id"])
                system_prompt = st.session_state.get("custom_system_prompt", DEFAULT_SYSTEM_PROMPT)
                temperature = st.session_state.get("temperature", 0.7)
                optimize_code = st.session_state.get("optimize_code", True)
                
                # Użyj funkcji optymalizującej kontekst
                optimized_messages = prepare_messages_with_token_management(
                    all_messages, 
                    system_prompt, 
                    model, 
                    llm_service
                )
                
                # Inicjalizuj sesję streamowania
                st.session_state["current_stream_response"] = ""
                st.session_state["code_blocks"] = {}
                st.session_state["stream_was_interrupted"] = False
                
                metadata = None
                
                try:
                    for chunk in llm_service.call_llm_streaming(
                        messages=optimized_messages,
                        model=model,
                        system_prompt=None,  # System prompt jest już dodany w optimized_messages
                        temperature=temperature,
                        max_tokens=12000,
                        optimize_code=optimize_code
                    ):
                        # Sprawdź, czy to obiekt błędu
                        if isinstance(chunk, dict) and "error" in chunk and chunk["error"]:
                            st.error(chunk["error_message"])
                            full_response = chunk.get("partial_response", "")
                            code_blocks = chunk.get("code_blocks", {})
                            st.session_state["current_stream_response"] = full_response
                            st.session_state["code_blocks"] = code_blocks
                            st.session_state["stream_was_interrupted"] = True
                            
                            # Przetwórz pełną odpowiedź, aby zastąpić markery bloków kodu
                            for block_id, block_data in code_blocks.items():
                                marker = f"```{block_data['lang']}\n[CODE_BLOCK_{block_id}]\n```"
                                code = block_data["code"]
                                
                                # Formatuj kod Python za pomocą black
                                if block_data['lang'] == "python" and optimize_code and len(code.splitlines()) > 10:
                                    try:
                                        code = format_code_with_black(code)
                                    except:
                                        pass
                                
                                full_response = full_response.replace(marker, f"```{block_data['lang']}\n{code}\n```")
                            
                            response_placeholder.markdown(full_response)
                            break
                        
                        # Sprawdź, czy to ostatni element z metadanymi
                        if isinstance(chunk, dict) and "full_response" in chunk:
                            metadata = chunk
                            full_response = metadata["full_response"] 
                            code_blocks = metadata.get("code_blocks", {})
                            break
                        
                        # Aktualizuj wyświetlanie na żywo
                        if not isinstance(chunk, dict):
                            # Pobierz aktualną pełną odpowiedź i bloki kodu ze stanu sesji
                            full_response = st.session_state.get("current_stream_response", "")
                            code_blocks = st.session_state.get("code_blocks", {})
                            
                            # Przetwórz pełną odpowiedź, aby zastąpić markery bloków kodu
                            display_response = full_response
                            for block_id, block_data in code_blocks.items():
                                marker = f"```{block_data['lang']}\n[CODE_BLOCK_{block_id}]\n```"
                                if marker in display_response:
                                    display_response = display_response.replace(marker, f"```{block_data['lang']}\n{block_data['code']}\n```")
                            
                            # Wyświetl odpowiedź
                            response_placeholder.markdown(display_response + "▌")
                        
                except Exception as e:
                    # Przechwycenie błędów streamowania
                    st.error(f"Błąd podczas streamowania: {str(e)}")
                    st.session_state["stream_was_interrupted"] = True
                    response_placeholder.markdown(full_response)
                    return
                
                # Przetwórz pełną odpowiedź, zastępując markery bloków kodu
                if metadata:
                    full_response = metadata["full_response"]
                    code_blocks = metadata.get("code_blocks", {})
                
                processed_response = full_response
                for block_id, block_data in code_blocks.items():
                    marker = f"```{block_data['lang']}\n[CODE_BLOCK_{block_id}]\n```"
                    code = block_data["code"]
                    
                    # Formatuj kod Python za pomocą black
                    if block_data['lang'] == "python" and optimize_code and len(code.splitlines()) > 10:
                        try:
                            code = format_code_with_black(code)
                        except:
                            pass
                    
                    processed_response = processed_response.replace(marker, f"```{block_data['lang']}\n{code}\n```")
                
                # Finalne wyświetlenie odpowiedzi bez kursora
                response_placeholder.markdown(processed_response)
                
                # Aktualizuj statystyki tokenów
                if metadata and "usage" in metadata:
                    usage = metadata["usage"]
                    st.session_state["token_usage"]["prompt"] += usage["prompt_tokens"]
                    st.session_state["token_usage"]["completion"] += usage["completion_tokens"]
                    
                    # Oblicz koszt
                    cost = calculate_cost(
                        model, 
                        usage["prompt_tokens"], 
                        usage["completion_tokens"]
                    )
                    
                    st.session_state["token_usage"]["cost"] += cost
                
                # Zapisz odpowiedź asystenta do bazy danych
                db.save_message(current_conversation_id, "assistant", processed_response)
                
                # Wyczyść pamięć streamowania po pomyślnym zapisaniu
                st.session_state["current_stream_response"] = ""
                st.session_state["code_blocks"] = {}
                
                # Odśwież stronę aby wyświetlić nową wiadomość bez spinnerów
                st.rerun()
                
            except Exception as e:
                # Zapisz stan błędu i częściową odpowiedź
                st.session_state["stream_was_interrupted"] = True
                
                st.error(f"Wystąpił błąd: {str(e)}")
                st.code(traceback.format_exc())
                
                # Jeśli mamy częściową odpowiedź, wyświetl ją
                if full_response:
                    st.warning("Częściowa odpowiedź przed wystąpieniem błędu:")
                    
                    # Przetwórz odpowiedź aby zastąpić markery bloków kodu
                    processed_response = full_response
                    for block_id, block_data in code_blocks.items():
                        marker = f"```{block_data['lang']}\n[CODE_BLOCK_{block_id}]\n```"
                        code = block_data["code"]
                        
                        # Formatuj kod Python za pomocą black
                        if block_data['lang'] == "python" and optimize_code and len(code.splitlines()) > 10:
                            try:
                                code = format_code_with_black(code)
                            except:
                                pass
                        
                        processed_response = processed_response.replace(marker, f"```{block_data['lang']}\n{code}\n```")
                    
                    st.markdown(processed_response)
                    st.session_state["current_stream_response"] = full_response
                    
                    if st.button("Zapisz tę częściową odpowiedź"):
                        db.save_message(current_conversation_id, "assistant", processed_response)
                        st.session_state["current_stream_response"] = ""
                        st.session_state["code_blocks"] = {}
                        st.success("Zapisano częściową odpowiedź!")
                        st.rerun()

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
