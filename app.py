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
import io
import traceback

# === Konfiguracja ===
MODEL_OPTIONS = [
    {
        "id": "anthropic/claude-3.7-sonnet:floor",
        "name": "Claude 3.7 Sonnet",
        "pricing": {"prompt": 3.0, "completion": 15.0},
        "description": "Zalecany - Najnowszy model Claude z doskonałymi umiejętnościami kodowania"
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

DEFAULT_SYSTEM_PROMPT = """Jesteś ekspertkim asystentem specjalizującym się w tworzeniu aplikacji Streamlit wykorzystujących AI. 
Pomagasz projektować, kodować i optymalizować aplikacje Streamlit, szczególnie te korzystające z modeli językowych i innych usług AI.

Twoja wiedza specjalistyczna obejmuje:
1. Pisanie czystego, efektywnego kodu Streamlit
2. Projektowanie skutecznych interfejsów użytkownika wykorzystujących AI
3. Integrację z API jak OpenRouter, OpenAI, Anthropic, itp.
4. Optymalizację wydajności i kosztów przy korzystaniu z usług AI
5. Wdrażanie najlepszych praktyk dla aplikacji Streamlit

Gdy podajesz przykłady kodu, przestrzegaj tych zasad:
- Dołączaj kompletne, działające rozwiązania, które można skopiować i użyć bezpośrednio
- Dodawaj krótkie komentarze wyjaśniające złożone części
- Formatuj kod z odpowiednim wcięciem i strukturą
- Skup się na najlepszych praktykach i efektywnych wzorcach Streamlit

Zawsze dziel aplikacje na logiczne komponenty i funkcje, zamiast pisać wszystko w jednym bloku kodu.
Pamiętaj o zarządzaniu stanem sesji w Streamlit i optymalizacji kosztów przy korzystaniu z API modeli językowych.
"""

# === Funkcja dekoratora dla autoryzacji ===
def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not st.session_state.get("authenticated", False):
            return login_page()
        return f(*args, **kwargs)
    return decorated

def login_page():
    """Strona logowania"""
    st.title("🔒 Logowanie")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Zaloguj się, aby uzyskać dostęp")
        
        username = st.text_input("Nazwa użytkownika", key="login_username")
        password = st.text_input("Hasło", type="password", key="login_password")
        
        if st.button("Zaloguj"):
            # Pobierz ustawienia z secrets
            correct_username = st.secrets.get("APP_USER", "admin")
            correct_password = st.secrets.get("APP_PASSWORD", "password")
            
            if username == correct_username and password == correct_password:
                st.session_state["authenticated"] = True
                st.success("Zalogowano pomyślnie!")
                st.rerun()
            else:
                st.error("Nieprawidłowa nazwa użytkownika lub hasło!")

# === Zarządzanie bazą danych ===
class AssistantDB:
    def __init__(self, db_path='streamlit_assistant.db'):
        """Inicjalizacja połączenia z bazą danych i tabel"""
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
    
    def _create_tables(self):
        """Utwórz tabele bazy danych, jeśli nie istnieją"""
        cursor = self.conn.cursor()
        
        # Tabela konwersacji
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        )
        ''')
        
        # Tabela wiadomości
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TIMESTAMP,
            attachments TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations (id)
        )
        ''')
        
        self.conn.commit()

    def save_message(self, conversation_id: str, role: str, content: str, attachments=None) -> str:
        """Zapisz wiadomość w bazie danych"""
        cursor = self.conn.cursor()
        message_id = str(uuid.uuid4())
        
        # Przygotuj załączniki do zapisu (konwersja danych binarnych)
        serializable_attachments = []
        if attachments:
            for attachment in attachments:
                # Tworzymy nowy słownik zawierający tylko serializowalne dane
                serialized = {
                    "type": attachment.get("type", ""),
                    "name": attachment.get("name", "")
                }
                
                # Jeśli jest text_content, dodajemy go
                if "text_content" in attachment:
                    serialized["text_content"] = attachment["text_content"]
                
                # Nie zapisujemy binarnych danych w bazie danych
                serializable_attachments.append(serialized)
        
        attachments_json = json.dumps(serializable_attachments)
        
        cursor.execute(
            "INSERT INTO messages (id, conversation_id, role, content, timestamp, attachments) VALUES (?, ?, ?, ?, ?, ?)",
            (message_id, conversation_id, role, content, datetime.now(), attachments_json)
        )
        self.conn.commit()
        return message_id
    
    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Pobierz wszystkie wiadomości dla konwersacji"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT role, content, attachments FROM messages WHERE conversation_id = ? ORDER BY timestamp",
            (conversation_id,)
        )
        
        messages = []
        for role, content, attachments_json in cursor.fetchall():
            try:
                attachments = json.loads(attachments_json) if attachments_json else []
                
                messages.append({
                    "role": role,
                    "content": content,
                    "attachments": attachments
                })
            except Exception as e:
                # W przypadku błędu, dodaj wiadomość bez załączników
                messages.append({
                    "role": role,
                    "content": content,
                    "attachments": []
                })
                
        return messages

    def get_messages_with_pagination(self, conversation_id: str, limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        """Pobierz wiadomości dla konwersacji z paginacją"""
        cursor = self.conn.cursor()
        
        if limit is not None:
            cursor.execute(
                "SELECT role, content, attachments FROM messages WHERE conversation_id = ? ORDER BY timestamp LIMIT ? OFFSET ?",
                (conversation_id, limit, offset)
            )
        else:
            cursor.execute(
                "SELECT role, content, attachments FROM messages WHERE conversation_id = ? ORDER BY timestamp",
                (conversation_id,)
            )
        
        messages = []
        for role, content, attachments_json in cursor.fetchall():
            try:
                attachments = json.loads(attachments_json) if attachments_json else []
                
                messages.append({
                    "role": role,
                    "content": content,
                    "attachments": attachments
                })
            except Exception as e:
                print(f"Błąd przetwarzania załącznika: {str(e)}")
                messages.append({
                    "role": role,
                    "content": content,
                    "attachments": []
                })
                
        return messages

    def get_message_count(self, conversation_id: str) -> int:
        """Pobierz liczbę wiadomości w konwersacji"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
            (conversation_id,)
        )
        return cursor.fetchone()[0]

    def save_conversation(self, conversation_id: str, title: str):
        """Utwórz lub zaktualizuj konwersację"""
        cursor = self.conn.cursor()
        now = datetime.now()
        
        # Sprawdź, czy konwersacja istnieje
        cursor.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,))
        if cursor.fetchone():
            # Aktualizuj istniejącą konwersację
            cursor.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, conversation_id)
            )
        else:
            # Utwórz nową konwersację
            cursor.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conversation_id, title, now, now)
            )
        
        self.conn.commit()
    
    def get_conversations(self) -> List[Dict[str, Any]]:
        """Pobierz wszystkie konwersacje"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, title, created_at FROM conversations ORDER BY updated_at DESC"
        )
        return [
            {"id": conv_id, "title": title, "created_at": created_at} 
            for conv_id, title, created_at in cursor.fetchall()
        ]

    def delete_conversation(self, conversation_id: str):
        """Usuń konwersację i jej wiadomości"""
        cursor = self.conn.cursor()
        # Najpierw usuń wiadomości (ograniczenie klucza obcego)
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        # Usuń konwersację
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        self.conn.commit()

# === Funkcje pomocnicze dla zarządzania kontekstem i załącznikami ===
def prepare_messages_with_token_management(messages, system_prompt, model_id, llm_service):
    """Przygotowuje wiadomości do wysłania, zarządzając limitami tokenów"""
    
    # Ustal limit tokenów dla różnych modeli
    model_token_limits = {
        "anthropic/claude-3.7-sonnet:floor": 180000,
        "anthropic/claude-3.7-sonnet:thinking": 180000,
        "anthropic/claude-3.5-haiku:floor": 150000,
        "openai/gpt-4o:floor": 120000, 
        "openai/gpt-4-turbo:floor": 100000,
    }
    
    # Domyślny limit, jeśli model nie jest znany
    default_token_limit = 100000
    max_input_tokens = model_token_limits.get(model_id, default_token_limit)
    
    # Rezerwuj tokeny na odpowiedź i prompt systemowy
    max_completion_tokens = 12000
    system_tokens = llm_service.count_tokens(system_prompt) if system_prompt else 0
    available_tokens = max_input_tokens - system_tokens - max_completion_tokens - 100  # 100 jako bufor bezpieczeństwa
    
    # Przygotuj wiadomości API
    api_messages = []
    if system_prompt:
        api_messages.append({"role": "system", "content": system_prompt})
    
    # Policz tokeny aktualnych wiadomości
    current_tokens = 0
    user_messages = []
    assistant_messages = []
    
    # Najpierw zbierz wszystkie wiadomości
    for msg in messages:
        if msg["role"] == "user":
            user_messages.append(msg)
        else:
            assistant_messages.append(msg)
    
    # Zawsze dołącz ostatnią wiadomość użytkownika
    if user_messages:
        last_user_message = user_messages.pop()
    else:
        last_user_message = None
    
    # Zawsze dołącz ostatnią odpowiedź asystenta, jeśli istnieje
    if assistant_messages:
        last_assistant_message = assistant_messages.pop()
    else:
        last_assistant_message = None
    
    # Pierwsza wiadomość użytkownika jako kontekst, jeśli istnieje
    if user_messages:
        first_user_message = user_messages.pop(0)
    else:
        first_user_message = None
    
    # Dodaj pierwszą wiadomość użytkownika do kontekstu
    if first_user_message:
        first_msg_tokens = llm_service.count_tokens(first_user_message["content"])
        if current_tokens + first_msg_tokens <= available_tokens:
            api_messages.append(first_user_message)
            current_tokens += first_msg_tokens
    
    # Proces dodawania pozostałych wiadomości w parach (zachowuje przepływ konwersacji)
    remaining_messages = []
    # Łączymy wiadomości z obu list naprzemiennie, zachowując kolejność
    i, j = 0, 0
    while i < len(user_messages) or j < len(assistant_messages):
        if i < len(user_messages):
            remaining_messages.append(user_messages[i])
            i += 1
        if j < len(assistant_messages):
            remaining_messages.append(assistant_messages[j])
            j += 1
    
    # Sortuj pozostałe wiadomości według czasu (najstarsze pierwsze)
    # Zakładamy, że wiadomości są już uporządkowane chronologicznie w bazie danych
    
    # Dodaj tyle wiadomości, ile zmieści się w limicie tokenów
    for msg in remaining_messages:
        msg_tokens = llm_service.count_tokens(msg["content"])
        
        # Jeśli wiadomość nie zmieści się, przerwij
        if current_tokens + msg_tokens > available_tokens:
            break
        
        api_messages.append(msg)
        current_tokens += msg_tokens
    
    # Zawsze dodaj ostatnią odpowiedź asystenta, jeśli istnieje i jest miejsce
    if last_assistant_message:
        last_assistant_tokens = llm_service.count_tokens(last_assistant_message["content"])
        if current_tokens + last_assistant_tokens <= available_tokens:
            api_messages.append(last_assistant_message)
            current_tokens += last_assistant_tokens
        else:
            # Jeśli nie zmieści się cała, dodaj skróconą wersję
            truncated_content = "POPRZEDNIA ODPOWIEDŹ (skrócona): " + last_assistant_message["content"][:1000] + "..."
            truncated_tokens = llm_service.count_tokens(truncated_content)
            if current_tokens + truncated_tokens <= available_tokens:
                api_messages.append({"role": "assistant", "content": truncated_content})
                current_tokens += truncated_tokens
    
    # Zawsze dodaj ostatnią wiadomość użytkownika
    if last_user_message:
        last_user_tokens = llm_service.count_tokens(last_user_message["content"])
        
        # Jeśli ostatnia wiadomość użytkownika jest za długa, spróbuj ją skrócić
        if current_tokens + last_user_tokens > available_tokens:
            # Oblicz, ile tokenów możemy użyć
            remaining_tokens = available_tokens - current_tokens
            
            # Jeśli za mało miejsca, dodaj tylko treść bez załączników
            if "attachments" in last_user_message and last_user_message["attachments"]:
                # Wyodrębnij treść bez załączników
                main_content = last_user_message["content"].split("\n\n[Załącznik")[0]
                main_content_tokens = llm_service.count_tokens(main_content)
                
                if main_content_tokens <= remaining_tokens:
                    # Dodaj tylko główną treść
                    modified_message = {"role": "user", "content": main_content}
                    api_messages.append(modified_message)
                else:
                    # Skróć nawet główną treść, jeśli potrzeba
                    truncated_content = main_content[:remaining_tokens * 4]  # Przybliżenie
                    modified_message = {"role": "user", "content": truncated_content}
                    api_messages.append(modified_message)
            else:
                # Skróć wiadomość użytkownika, jeśli nie ma załączników
                truncated_content = last_user_message["content"][:remaining_tokens * 4]  # Przybliżenie
                modified_message = {"role": "user", "content": truncated_content}
                api_messages.append(modified_message)
        else:
            # Dodaj pełną wiadomość
            api_messages.append(last_user_message)
    
    return api_messages

def process_attachments(attachments):
    """Przetwarza załączniki w format odpowiedni do wysłania do API"""
    processed_content = ""
    
    for attachment in attachments:
        att_type = attachment.get("type")
        att_name = attachment.get("name", "")
        
        if att_type == "file" and "text_content" in attachment:
            # Sprawdź, czy zawartość to blok kodu
            if attachment['text_content'].startswith("```") and attachment['text_content'].endswith("```"):
                processed_content += f"\n\n[KOD: {att_name}]\n{attachment['text_content']}\n"
            else:
                processed_content += f"\n\n[PLIK: {att_name}]\n{attachment['text_content']}\n"
    
    return processed_content.strip()

def display_code_block(code, language="python"):
    """Wyświetla blok kodu z opcjami kopiowania i pobrania"""
    st.code(code, language=language)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Kopiuj kod", key=f"copy_{hash(code)}", use_container_width=True):
            st.code(code)
            st.info("Skopiuj powyższy kod")
    
    with col2:
        # Generuj plik do pobrania
        if st.button("Pobierz plik", key=f"download_{hash(code)}", use_container_width=True):
            filename = f"code_{language}.{language}"
            st.download_button(
                label="Pobierz", 
                data=code, 
                file_name=filename,
                mime="text/plain",
                key=f"dl_{hash(code)}"
            )

# === Serwis LLM ===
class LLMService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        
        # Prosta pamięć podręczna dla powtarzających się pytań
        self.cache = {}

    def count_tokens(self, text: str) -> int:
        """Oszacuj liczbę tokenów dla Claude"""
        try:
            encoding = tiktoken.encoding_for_model("gpt-4")  # Używamy kodowania gpt-4 jako przybliżenia
            return len(encoding.encode(text))
        except:
            # Fallback do cl100k_base, jeśli określone kodowanie nie jest dostępne
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))
            except Exception as e:
                # Ostateczny fallback - prymitywne oszacowanie
                return len(text) // 4  # Bardzo zgrubne przybliżenie: ~4 znaki na token

    def get_cache_key(self, messages, model, system_prompt, temperature):
        """Generuj klucz pamięci podręcznej dla zapytania LLM"""
        cache_input = f"{json.dumps(messages)}-{model}-{system_prompt}-{temperature}"
        return hashlib.md5(cache_input.encode()).hexdigest()

    def call_llm(self, 
                messages: List[Dict[str, Any]], 
                model: str = "anthropic/claude-3.7-sonnet:floor", 
                system_prompt: str = None, 
                temperature: float = 0.7, 
                max_tokens: int = 12000,
                use_cache: bool = True) -> Dict[str, Any]:
        """Wywołaj API LLM przez OpenRouter"""
        # Sprawdź pamięć podręczną, jeśli używamy cachowania
        if use_cache and temperature < 0.1:  # Cachujemy tylko deterministyczne odpowiedzi
            cache_key = self.get_cache_key(messages, model, system_prompt, temperature)
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        # Przygotuj wiadomości z promptem systemowym, jeśli podano
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        
        # Dodaj wszystkie wiadomości
        for msg in messages:
            api_messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Oblicz tokeny promptu
        prompt_text = system_prompt or ""
        for msg in messages:
            if isinstance(msg.get("content"), str):
                prompt_text += msg["content"]
        
        prompt_tokens = self.count_tokens(prompt_text)
        
        # Zdefiniuj parametry ponawiania
        max_retries = 3
        retry_delay = 2  # sekundy
        
        # Próba wywołania API z ponawianiem
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
                    timeout=300  # Zwiększenie timeout do 5 minut
                )
                response.raise_for_status()
                result = response.json()
                
                # Oszacuj tokeny odpowiedzi
                response_content = result["choices"][0]["message"]["content"]
                completion_tokens = self.count_tokens(response_content)
                
                # Dodaj informacje o tokenach do wyniku
                result["usage"] = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
                
                # Dodaj do pamięci podręcznej, jeśli używamy cachowania
                if use_cache and temperature < 0.1:
                    cache_key = self.get_cache_key(messages, model, system_prompt, temperature)
                    self.cache[cache_key] = result
                
                return result
            
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:  # Ostatnia próba
                    raise Exception(f"Nie udało się połączyć z API po {max_retries} próbach: {str(e)}")
                
                # Czekaj przed ponowieniem
                time.sleep(retry_delay * (2 ** attempt))  # Wykładnicze wycofanie
    
    def call_llm_streaming(self, 
                         messages: List[Dict[str, Any]], 
                         model: str = "anthropic/claude-3.7-sonnet:floor", 
                         system_prompt: str = None, 
                         temperature: float = 0.7, 
                         max_tokens: int = 12000) -> Generator[str, None, None]:
        """Wywołaj API LLM przez OpenRouter ze streamingiem odpowiedzi"""
        
        # Przygotuj wiadomości z promptem systemowym, jeśli podano
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        
        # Dodaj wszystkie wiadomości
        for msg in messages:
            api_messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Oblicz tokeny promptu
        prompt_text = system_prompt or ""
        for msg in messages:
            if isinstance(msg.get("content"), str):
                prompt_text += msg["content"]
        
        prompt_tokens = self.count_tokens(prompt_text)
        
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
            timeout=300  # Zwiększenie timeout do 5 minut
        )
        
        # Przygotuj zmienne do śledzenia odpowiedzi
        full_response = ""
        completion_tokens = 0
        
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
                                full_response += content_chunk
                                completion_tokens += self.count_tokens(content_chunk)
                                yield content_chunk
                    except json.JSONDecodeError:
                        pass
        
        # Dodajemy metadane na końcu, aby mogły zostać wykorzystane przez wywołujący kod
        yield {
            "full_response": full_response,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

# === Funkcje pomocnicze ===
def calculate_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Oblicz szacowany koszt zapytania w USD"""
    for model in MODEL_OPTIONS:
        if model["id"] == model_id:
            return (prompt_tokens / 1_000_000) * model["pricing"]["prompt"] + \
                   (completion_tokens / 1_000_000) * model["pricing"]["completion"]
    return 0.0  # W przypadku nieznalezienia modelu

def format_message_for_display(message: Dict[str, str]) -> str:
    """Formatuj wiadomość do wyświetlenia w interfejsie, ze wsparciem dla bloków kodu"""
    content = message.get("content", "")
    
    # Wyodrębnianie i formatowanie bloków kodu
    def replace_code_block(match):
        lang = match.group(1) or ""
        code = match.group(2)
        return f"```{lang}\n{code}\n```"
    
    # Zastąp bloki kodu ze składnią markdown
    content = re.sub(r"```(.*?)\n(.*?)```", replace_code_block, content, flags=re.DOTALL)
    
    return content

def get_conversation_title(messages: List[Dict[str, str]], llm_service: LLMService, api_key: str) -> str:
    """Wygeneruj tytuł dla nowej konwersacji na podstawie pierwszej wiadomości użytkownika"""
    if not messages:
        return f"Nowa konwersacja {datetime.now().strftime('%d-%m-%Y %H:%M')}"
    
    # Użyj pierwszej wiadomości użytkownika jako podstawy tytułu
    user_message = next((m["content"] for m in messages if m["role"] == "user"), "")
    
    if len(user_message) > 40:
        # Skorzystaj z LLM, aby stworzyć krótki, opisowy tytuł
        try:
            response = llm_service.call_llm(
                messages=[
                    {"role": "user", "content": f"Utwórz krótki, opisowy tytuł (max. 5 słów) dla następującej konwersacji, bez cudzysłowów: {user_message[:200]}..."}
                ],
                model="anthropic/claude-3.5-haiku:floor",  # Tańszy model jest wystarczający do tworzenia tytułów
                system_prompt="Jesteś pomocnym asystentem, który tworzy krótkie, opisowe tytuły konwersacji.",
                temperature=0.2,
                max_tokens=20,
                use_cache=True
            )
            title = response["choices"][0]["message"]["content"].strip().strip('"\'')
            # Usuń znaki, które mogłyby sprawiać problemy z interfejsem
            title = re.sub(r'[^\w\s\-.,]', '', title)
            return title[:40]
        except Exception:
            # W przypadku błędu, wróć do domyślnego tytułu
            pass
    
    # Domyślnie użyj skróconej wiadomości użytkownika
    return user_message[:40] + ("..." if len(user_message) > 40 else "")

def parse_code_blocks(content):
    """Wyodrębnij bloki kodu z treści markdown"""
    code_blocks = []
    pattern = r"```([a-zA-Z0-9]*)\n(.*?)\n```"
    matches = re.findall(pattern, content, re.DOTALL)
    
    for lang, code in matches:
        # Standardowy język, jeśli nie został określony
        language = lang.strip() if lang.strip() else "python"
        code_blocks.append({"language": language, "code": code})
    
    return code_blocks

# === Komponenty interfejsu użytkownika ===
def sidebar_component():
    """Komponent paska bocznego z konwersacjami i ustawieniami"""
    st.sidebar.title("AI Asystent Developera")
    
    # Ustawienia modelu
    with st.sidebar.expander("⚙️ Ustawienia modelu", expanded=False):
        model_options = {model["id"]: f"{model['name']}" for model in MODEL_OPTIONS}
        selected_model = st.selectbox(
            "Model LLM",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0,
            key="model_selection"
        )
        
        # Pokaż opis modelu
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
    
    # Statystyki tokenów
    if "token_usage" in st.session_state:
        with st.sidebar.expander("📊 Statystyki tokenów", expanded=False):
            st.metric("Tokeny prompt", st.session_state["token_usage"]["prompt"])
            st.metric("Tokeny completion", st.session_state["token_usage"]["completion"])
            st.metric("Szacunkowy koszt", f"${st.session_state['token_usage']['cost']:.4f}")
    
    # Lista konwersacji
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
        
        # Przycisk nowej konwersacji
        if st.sidebar.button("➕ Nowa konwersacja", use_container_width=True):
            st.session_state["current_conversation_id"] = str(uuid.uuid4())
            # Wyczyść załączniki dla nowej konwersacji
            st.session_state["attachments"] = []
            st.rerun()

@requires_auth
def chat_component():
    """Komponent interfejsu czatu"""
    # Dodaj niestandardowy CSS dla przypiętego paska wejściowego
    st.markdown("""
    <style>
    /* Miejsce na pasek wejściowy na dole */
    .main .block-container {
        padding-bottom: 80px;
    }
    
    /* Przytwierdzony pasek wejściowy na dole ekranu */
    .stChatInputContainer {
        position: fixed;
        bottom: 0;
        left: 240px; /* Miejsce na sidebar */
        right: 0;
        padding: 1rem;
        background: white;
        z-index: 999;
        border-top: 1px solid #ddd;
    }
    
    /* Stylowanie załączników */
    .attachment-badge {
        display: inline-block;
        padding: 2px 8px;
        margin: 2px;
        background-color: #f0f2f6;
        border-radius: 10px;
        font-size: 0.8em;
    }
    
    /* Style dla bloków kodu */
    .stMarkdown pre {
        overflow-x: auto;
    }
    
    /* Na urządzeniach mobilnych */
    @media (max-width: 768px) {
        .stChatInputContainer {
            left: 0;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Pobierz klucz API z secrets
    api_key = st.secrets.get("OPENROUTER_API_KEY", "")
    if not api_key:
        st.warning("⚠️ Brak klucza API OpenRouter w ustawieniach secrets.")
        return

    # Aktualizuj klucz API w sesji
    st.session_state["api_key"] = api_key

    # Pobierz instancję serwisu LLM i bazy danych
    if "llm_service" not in st.session_state or st.session_state.get("llm_service") is None:
        st.session_state["llm_service"] = LLMService(api_key)

    llm_service = st.session_state.get("llm_service")
    db = st.session_state.get("db")

    if not llm_service or not db:
        st.error("⚠️ Nie można zainicjalizować serwisów. Odśwież stronę i spróbuj ponownie.")
        return

    # Inicjalizacja zmiennych sesji dla konwersacji
    if "current_conversation_id" not in st.session_state:
        st.session_state["current_conversation_id"] = str(uuid.uuid4())

    current_conversation_id = st.session_state["current_conversation_id"]

    # Statystyki konwersacji
    if "token_usage" not in st.session_state:
        st.session_state["token_usage"] = {"prompt": 0, "completion": 0, "cost": 0.0}
    
    # Inicjalizacja zmiennych dla załączników
    if "attachments" not in st.session_state:
        st.session_state["attachments"] = []

    # Pobierz wiadomości
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
                # Sprawdź, czy odpowiedź zawiera bloki kodu do wyświetlenia
                code_blocks = parse_code_blocks(content)
                
                # Wyświetl treść ogólną (bez bloków kodu, które będą wyświetlane oddzielnie)
                clean_content = content
                for match in re.finditer(r"```.*?```", content, re.DOTALL):
                    start, end = match.span()
                    clean_content = clean_content.replace(match.group(), f"[BLOK KODU #{match.start()}]")
                
                st.markdown(clean_content)
                
                # Wyświetl każdy blok kodu z dodatkowymi kontrolkami
                if code_blocks:
                    st.write("### Bloki kodu:")
                    for i, block in enumerate(code_blocks):
                        with st.expander(f"Kod {i+1} - {block['language']}", expanded=True):
                            display_code_block(block['code'], block['language'])

    # Wyświetlanie załączników (jako tekst pod polem wejściowym)
    if st.session_state["attachments"]:
        attachment_text = "Załączniki: " + " ".join([
            f"<span class='attachment-badge'>{attachment.get('type')} | {attachment.get('name')[:15]}...</span>"
            for attachment in st.session_state["attachments"]
        ])
        st.markdown(f"<div style='margin-bottom: 5px'>{attachment_text}</div>", unsafe_allow_html=True)
    
    # Obsługa załączników - ikony pod polem czatu
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("📄 Plik", use_container_width=True):
            st.session_state["show_file_uploader"] = not st.session_state.get("show_file_uploader", False)
            st.rerun()
    
    with col2:
        if st.button("💻 Kod", use_container_width=True):
            st.session_state["show_code_input"] = not st.session_state.get("show_code_input", False)
            st.rerun()
    
    # Zarządzanie istniejącymi załącznikami
    if st.session_state["attachments"]:
        cols = st.columns(len(st.session_state["attachments"]))
        for i, (col, attachment) in enumerate(zip(cols, st.session_state["attachments"])):
            with col:
                if st.button(f"❌ {attachment.get('name', '')[:7]}...", key=f"del_{i}"):
                    st.session_state["attachments"].pop(i)
                    st.rerun()

    # Formularze załączników
    if st.session_state.get("show_file_uploader", False):
        with st.expander("Dodaj plik tekstowy", expanded=True):
            uploaded_file = st.file_uploader("Wybierz plik", type=["txt", "md", "json", "csv", "py", "js", "html", "css"], key="text_upload")
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Anuluj", use_container_width=True):
                    st.session_state["show_file_uploader"] = False
                    st.rerun()
            with col2:
                if uploaded_file is not None and st.button("Dodaj", use_container_width=True):
                    try:
                        text_content = uploaded_file.getvalue().decode("utf-8")
                        st.session_state["attachments"].append({
                            "type": "file",
                            "name": uploaded_file.name,
                            "text_content": text_content
                        })
                        st.session_state["show_file_uploader"] = False
                        st.success(f"Dodano plik: {uploaded_file.name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Błąd odczytu pliku: {str(e)}")

    if st.session_state.get("show_code_input", False):
        with st.expander("Dodaj kod", expanded=True):
            code_language = st.selectbox("Język programowania", 
                                         ["python", "javascript", "html", "css", "json", "sql", "bash"], 
                                         key="code_language")
            code_content = st.text_area("Wklej kod", height=150, key="code_content")
            file_name = st.text_input("Nazwa pliku (opcjonalnie)", 
                                      value=f"code.{code_language}", 
                                      key="code_filename")

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Anuluj", use_container_width=True):
                    st.session_state["show_code_input"] = False
                    st.rerun()
            with col2:
                if code_content and st.button("Dodaj", use_container_width=True):
                    st.session_state["attachments"].append({
                        "type": "file",
                        "name": file_name,
                        "text_content": f"```{code_language}\n{code_content}\n```"
                    })
                    st.session_state["show_code_input"] = False
                    st.success(f"Dodano kod: {file_name}")
                    st.rerun()
    
    # Pole wejściowe użytkownika - ostatni element
    user_input = st.chat_input("Wpisz swoje pytanie lub zadanie...")

    # Obsługa wprowadzonego komunikatu
    if user_input:
        # Natychmiast wyświetl wiadomość użytkownika
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Przygotuj treść wiadomości i załączniki
        message_content = user_input
        
        # Przetwórz załączniki
        attachments_to_send = st.session_state.get("attachments", []).copy()
        
        # Dodaj informacje o załącznikach do treści wiadomości
        attachments_text = process_attachments(attachments_to_send)
        if attachments_text:
            message_content += "\n\n" + attachments_text
        
        # Sprawdź, czy konwersacja ma tytuł
        if len(messages) == 0:
            conversation_title = get_conversation_title([{"role": "user", "content": user_input}], llm_service, st.session_state["api_key"])
            db.save_conversation(current_conversation_id, conversation_title)
        
        # Zapisz wiadomość użytkownika w bazie danych
        db.save_message(current_conversation_id, "user", user_input, attachments_to_send)
        
        # Pobierz wszystkie wiadomości, aby zachować kontekst
        all_messages = db.get_messages(current_conversation_id)
        
        # Wyświetl oczekującą odpowiedź asystenta
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                model = st.session_state.get("model_selection", MODEL_OPTIONS[0]["id"])
                system_prompt = st.session_state.get("custom_system_prompt", DEFAULT_SYSTEM_PROMPT)
                temperature = st.session_state.get("temperature", 0.7)
                
                # Użyj funkcji optymalizującej kontekst
                optimized_messages = prepare_messages_with_token_management(
                    all_messages, 
                    system_prompt, 
                    model, 
                    llm_service
                )
                
                # Wywołaj API ze streamingiem
                metadata = None
                for chunk in llm_service.call_llm_streaming(
                    messages=optimized_messages,
                    model=model,
                    system_prompt=None,  # System prompt jest już dodany w optimized_messages
                    temperature=temperature,
                    max_tokens=12000
                ):
                    # Sprawdź, czy to ostatni element z metadanymi
                    if isinstance(chunk, dict) and "full_response" in chunk:
                        metadata = chunk
                        break
                    
                    # Aktualizuj odpowiedź w czasie rzeczywistym
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                # Finalne wyświetlenie odpowiedzi bez kursora
                response_placeholder.markdown(full_response)
                
                # Aktualizuj statystyki tokenów, jeśli mamy metadane
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
                
                # Wyświetl bloki kodu oddzielnie, jeśli istnieją
                code_blocks = parse_code_blocks(full_response)
                if code_blocks:
                    st.write("### Bloki kodu:")
                    for i, block in enumerate(code_blocks):
                        with st.expander(f"Kod {i+1} - {block['language']}", expanded=True):
                            display_code_block(block['code'], block['language'])
                
                # Zapisz odpowiedź asystenta
                db.save_message(current_conversation_id, "assistant", full_response)
                
                # Wyczyść załączniki po wysłaniu
                st.session_state["attachments"] = []
                # Ukryj formularze załączników
                st.session_state["show_file_uploader"] = False  
                st.session_state["show_code_input"] = False
                
                # Odśwież stronę aby wyświetlić nową wiadomość bez spinnerów
                st.rerun()
                
            except Exception as e:
                st.error(f"Wystąpił błąd: {str(e)}")
                st.code(traceback.format_exc())

# === Główna aplikacja ===
def main():
    st.set_page_config(
        page_title="AI Asystent Developera Streamlit",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inicjalizacja serwisów
    if "db" not in st.session_state:
        st.session_state["db"] = AssistantDB()
    
    # Sprawdź autentykację
    if not st.session_state.get("authenticated", False):
        login_page()
        return
    
    # Pobierz klucz API z secrets
    api_key = st.secrets.get("OPENROUTER_API_KEY", "")
    if api_key and "llm_service" not in st.session_state:
        st.session_state["llm_service"] = LLMService(api_key)
        st.session_state["api_key"] = api_key
    
    # Wyświetl sidebar
    sidebar_component()
    
    # Wyświetl główny komponent czatu
    chat_component()

if __name__ == "__main__":
    main()
