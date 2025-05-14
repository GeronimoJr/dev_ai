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
from typing import List, Dict, Any, Optional
import hashlib
from functools import wraps

# === Konfiguracja ===
KNOWLEDGE_CATEGORIES = [
    "Komponenty UI",
    "Integracja API",
    "Implementacja AI",
    "Przetwarzanie danych",
    "Optymalizacja wydajności",
    "Ogólne"
]

MODEL_OPTIONS = [
    {
        "id": "anthropic/claude-3.7-sonnet",
        "name": "Claude 3.7 Sonnet",
        "pricing": {"prompt": 3.0, "completion": 15.0},
        "description": "Zalecany - Najnowszy model Claude z doskonałymi umiejętnościami kodowania"
    },
    {
        "id": "anthropic/claude-3.7-haiku-thinking",
        "name": "Claude 3.7 Thinking",
        "pricing": {"prompt": 3.0, "completion": 15.0},
        "description": "Model Claude wykorzystujący dodatkowy czas na analizę problemów"
    },
    {
        "id": "openai/gpt-4o",
        "name": "GPT-4o",
        "pricing": {"prompt": 2.5, "completion": 10.0},
        "description": "Silna alternatywa z dobrymi zdolnościami kodowania"
    },
    {
        "id": "openai/gpt-4-turbo",
        "name": "GPT-4 Turbo",
        "pricing": {"prompt": 2.5, "completion": 10.0},
        "description": "Nieco starszy model GPT-4 Turbo"
    },
    {
        "id": "anthropic/claude-3.5-haiku",
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
:floordevai.txt:
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
        
        # Tabela bazy wiedzy
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge_items (
            id TEXT PRIMARY KEY,
            title TEXT,
            content TEXT,
            category TEXT,
            tags TEXT,
            created_at TIMESTAMP
        )
        ''')
        
        self.conn.commit()

    def save_message(self, conversation_id: str, role: str, content: str, attachments: List[Dict] = None) -> str:
        """Zapisz wiadomość w bazie danych"""
        cursor = self.conn.cursor()
        message_id = str(uuid.uuid4())
        attachments_json = json.dumps(attachments or [])
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
        return [
            {
                "role": role, 
                "content": content, 
                "attachments": json.loads(attachments) if attachments else []
            } 
            for role, content, attachments in cursor.fetchall()
        ]

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

    def save_knowledge_item(self, title: str, content: str, category: str, tags: List[str] = None) -> str:
        """Zapisz element w bazie wiedzy"""
        cursor = self.conn.cursor()
        item_id = str(uuid.uuid4())
        tags_json = json.dumps(tags or [])
        
        cursor.execute(
            "INSERT INTO knowledge_items (id, title, content, category, tags, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (item_id, title, content, category, tags_json, datetime.now())
        )
        
        self.conn.commit()
        return item_id

    def get_knowledge_items(self, category: str = None) -> List[Dict[str, Any]]:
        """Pobierz wszystkie elementy bazy wiedzy, opcjonalnie filtrowane według kategorii"""
        cursor = self.conn.cursor()
        
        if category and category != "Wszystkie":
            cursor.execute(
                "SELECT id, title, content, category, tags FROM knowledge_items WHERE category = ? ORDER BY created_at DESC",
                (category,)
            )
        else:
            cursor.execute(
                "SELECT id, title, content, category, tags FROM knowledge_items ORDER BY created_at DESC"
            )
        
        return [
            {
                "id": item_id,
                "title": title,
                "content": content,
                "category": category,
                "tags": json.loads(tags)
            }
            for item_id, title, content, category, tags in cursor.fetchall()
        ]

    def search_knowledge_base(self, query: str, category: str = None) -> List[Dict[str, Any]]:
        """Przeszukaj bazę wiedzy według tekstu zapytania"""
        cursor = self.conn.cursor()
        search_term = f"%{query}%"
        
        if category and category != "Wszystkie":
            cursor.execute(
                "SELECT id, title, content, category, tags FROM knowledge_items WHERE (title LIKE ? OR content LIKE ?) AND category = ? ORDER BY created_at DESC",
                (search_term, search_term, category)
            )
        else:
            cursor.execute(
                "SELECT id, title, content, category, tags FROM knowledge_items WHERE title LIKE ? OR content LIKE ? ORDER BY created_at DESC",
                (search_term, search_term)
            )
        
        return [
            {
                "id": item_id,
                "title": title,
                "content": content,
                "category": category,
                "tags": json.loads(tags)
            }
            for item_id, title, content, category, tags in cursor.fetchall()
        ]

    def delete_knowledge_item(self, item_id: str):
        """Usuń element bazy wiedzy"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM knowledge_items WHERE id = ?", (item_id,))
        self.conn.commit()

    def update_knowledge_item(self, item_id: str, title: str, content: str, category: str, tags: List[str] = None):
        """Aktualizuj element bazy wiedzy"""
        cursor = self.conn.cursor()
        tags_json = json.dumps(tags or [])
        
        cursor.execute(
            "UPDATE knowledge_items SET title = ?, content = ?, category = ?, tags = ? WHERE id = ?",
            (title, content, category, tags_json, item_id)
        )
        
        self.conn.commit()

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
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))

    def get_cache_key(self, messages, model, system_prompt, temperature):
        """Generuj klucz pamięci podręcznej dla zapytania LLM"""
        cache_input = f"{json.dumps(messages)}-{model}-{system_prompt}-{temperature}"
        return hashlib.md5(cache_input.encode()).hexdigest()

    def call_llm(self, 
                messages: List[Dict[str, str]], 
                model: str = "anthropic/claude-3.7-sonnet", 
                system_prompt: str = None, 
                temperature: float = 0.7, 
                max_tokens: int = 4000,
                use_cache: bool = True) -> Dict[str, Any]:
        """Wywołaj API LLM przez OpenRouter z opcjonalnym cachowaniem"""
        # Sprawdź pamięć podręczną, jeśli używamy cachowania
        if use_cache and temperature < 0.1:  # Cachujemy tylko deterministyczne odpowiedzi
            cache_key = self.get_cache_key(messages, model, system_prompt, temperature)
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        # Przygotuj wiadomości z promptem systemowym, jeśli podano
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        api_messages.extend(messages)
        
        # Oblicz tokeny promptu
        prompt_text = system_prompt or ""
        for msg in messages:
            prompt_text += msg["content"]
        prompt_tokens = self.count_tokens(prompt_text)
        
        # Zdefiniuj parametry ponawiania
        max_retries = 3
        retry_delay = 2  # sekundy
        
        # Próba wywołania API z ponawianiem
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model,
                        "messages": api_messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    },
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                
                # Oszacuj tokeny odpowiedzi
                response_content = result["choices"][0]["message"]["content"]
                completion_tokens = self.count_tokens(response_content)
                
                # Dodaj informacje o tokenach i tworzenach do wyniku
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
                model="anthropic/claude-3.5-haiku",  # Tańszy model jest wystarczający do tworzenia tytułów
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

# === Komponenty interfejsu użytkownika ===
def sidebar_component():
    """Komponent paska bocznego z konwersacjami i ustawieniami"""
    st.sidebar.title("AI Asystent Developera")
    
    # Opcje nawigacji
    page = st.sidebar.radio("Nawigacja", ["Chat Asystent", "Baza Wiedzy"])
    
    # Ustawienia modelu w sekcji Chat
    if page == "Chat Asystent":
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
                st.rerun()
    
    return page

@requires_auth
def chat_component():
    """Komponent interfejsu czatu"""
    st.title("💬 Chat Asystent Developera Streamlit")
    
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
    
    with st.expander("📊 Statystyki tokenów", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tokeny prompt", st.session_state["token_usage"]["prompt"])
        with col2:
            st.metric("Tokeny completion", st.session_state["token_usage"]["completion"])
        with col3:
            st.metric("Szacunkowy koszt", f"${st.session_state['token_usage']['cost']:.4f}")
    
    # Kontener na historię czatu z możliwością przewijania
    chat_container = st.container()
    
    # Kontener na pola wprowadzania (zawsze na dole)
    input_container = st.container()
    
    # Wyświetl istniejące wiadomości
    messages = db.get_messages(current_conversation_id)
    
    with chat_container:
        if not messages:
            # Komunikat powitalny dla nowej konwersacji
            st.markdown(""" 
            ### 👋 Witaj w Asystencie Developera Streamlit!
            
            Jestem tu, aby pomóc Ci projektować i tworzyć aplikacje Streamlit z wykorzystaniem AI. 
            Możesz zadać mi pytania dotyczące:
            
            - Projektowania interfejsu użytkownika w Streamlit
            - Implementacji funkcjonalności AI w aplikacjach
            - Integracji z APIami (OpenRouter, OpenAI, Anthropic, itp.)
            - Optymalizacji wydajności i kosztów
            - Przykładów kodu i komponentów
            
            Zacznij od opisania aplikacji, którą chcesz stworzyć lub zapytaj o konkretne rozwiązanie.
            """)
        else:
            for message in messages:
                role = message["role"]
                content = format_message_for_display(message)
                attachments = message.get("attachments", [])
                
                if role == "user":
                    with st.chat_message("user"):
                        st.markdown(content)
                        # Wyświetl załączniki
                        for attachment in attachments:
                            if attachment.get("type") == "image":
                                st.image(attachment.get("data"), caption=attachment.get("name", "Załącznik"))
                            elif attachment.get("type") == "file":
                                st.download_button(
                                    label=f"📎 {attachment.get('name', 'Pobierz załącznik')}",
                                    data=attachment.get("data"),
                                    file_name=attachment.get("name", "attachment.txt"),
                                )
                elif role == "assistant":
                    with st.chat_message("assistant"):
                        st.markdown(content)
    
    # Sekcja załączników i wiadomości na dole ekranu
    with input_container:
        # Kontener na załączniki
        if "attachments" not in st.session_state:
            st.session_state["attachments"] = []
        
        # Wyświetl aktualnie dodane załączniki
        if st.session_state["attachments"]:
            st.write("Załączniki do wysłania:")
            cols = st.columns(4)
            for i, attachment in enumerate(st.session_state["attachments"]):
                with cols[i % 4]:
                    st.write(f"📎 {attachment.get('name', 'Załącznik')}")
                    if st.button("Usuń", key=f"remove_{i}"):
                        st.session_state["attachments"].pop(i)
                        st.rerun()
        
        # Dodaj załączniki
        with st.expander("📎 Dodaj załącznik", expanded=False):
            attachment_type = st.radio("Typ załącznika", ["Obraz", "Plik tekstowy", "Kod"])
            
            if attachment_type == "Obraz":
                uploaded_file = st.file_uploader("Wybierz obraz", type=["png", "jpg", "jpeg"], key="image_upload")
                if uploaded_file is not None and st.button("Dodaj obraz"):
                    bytes_data = uploaded_file.getvalue()
                    st.session_state["attachments"].append({
                        "type": "image",
                        "name": uploaded_file.name,
                        "data": bytes_data
                    })
                    st.success(f"Dodano obraz: {uploaded_file.name}")
                    st.rerun()
            
            elif attachment_type == "Plik tekstowy":
                uploaded_file = st.file_uploader("Wybierz plik", type=["txt", "md", "json", "csv"], key="text_upload")
                if uploaded_file is not None and st.button("Dodaj plik"):
                    bytes_data = uploaded_file.getvalue()
                    try:
                        text_content = bytes_data.decode("utf-8")
                        st.session_state["attachments"].append({
                            "type": "file",
                            "name": uploaded_file.name,
                            "data": bytes_data,
                            "text_content": text_content
                        })
                        st.success(f"Dodano plik: {uploaded_file.name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Błąd odczytu pliku: {str(e)}")
            
            elif attachment_type == "Kod":
                code_language = st.selectbox("Język programowania", ["python", "javascript", "html", "css", "json", "sql", "bash"])
                code_content = st.text_area("Wklej kod", height=150)
                file_name = st.text_input("Nazwa pliku (opcjonalnie)", value=f"code.{code_language}")
                
                if code_content and st.button("Dodaj kod"):
                    st.session_state["attachments"].append({
                        "type": "file",
                        "name": file_name,
                        "data": code_content.encode("utf-8"),
                        "text_content": f"```{code_language}\n{code_content}\n```"
                    })
                    st.success(f"Dodano kod: {file_name}")
                    st.rerun()
        
        # Input użytkownika
        user_input = st.chat_input("Wpisz swoje pytanie lub zadanie...")
        
        if user_input:
            # Przygotuj treść wiadomości i załączniki
            message_content = user_input
            attachments_to_send = st.session_state.get("attachments", [])
            
            # Dodaj informacje o załącznikach do treści wiadomości
            if attachments_to_send:
                attachment_descriptions = []
                for attachment in attachments_to_send:
                    if attachment.get("type") == "image":
                        attachment_descriptions.append(f"[Załącznik obrazu: {attachment.get('name', 'image')}]")
                    elif attachment.get("type") == "file" and attachment.get("text_content"):
                        attachment_descriptions.append(f"[Załącznik pliku: {attachment.get('name', 'file')}]\n{attachment.get('text_content')}")
                
                if attachment_descriptions:
                    message_content += "\n\n" + "\n\n".join(attachment_descriptions)
            
            # Sprawdź, czy konwersacja ma tytuł
            if len(messages) == 0:
                conversation_title = get_conversation_title([{"role": "user", "content": user_input}], llm_service, st.session_state["api_key"])
                db.save_conversation(current_conversation_id, conversation_title)
            
            # Wyświetl wiadomość użytkownika
            with st.chat_message("user"):
                st.markdown(user_input)
                # Wyświetl załączniki
                for attachment in attachments_to_send:
                    if attachment.get("type") == "image":
                        st.image(attachment.get("data"), caption=attachment.get("name", "Załącznik"))
                    elif attachment.get("type") == "file":
                        st.download_button(
                            label=f"📎 {attachment.get('name', 'Pobierz załącznik')}",
                            data=attachment.get("data"),
                            file_name=attachment.get("name", "attachment.txt"),
                        )
            
            # Zapisz wiadomość użytkownika
            db.save_message(current_conversation_id, "user", user_input, attachments_to_send)
            
            # Przygotuj wiadomości dla API
            api_messages = db.get_messages(current_conversation_id)
            
            # Konwertuj format wiadomości na ten oczekiwany przez API
            api_messages_formatted = []
            for msg in api_messages:
                api_msg = {"role": msg["role"], "content": msg["content"]}
                api_messages_formatted.append(api_msg)
            
            # Uzyskaj odpowiedź asystenta
            with st.spinner("Generowanie odpowiedzi..."):
                try:
                    model = st.session_state.get("model_selection", MODEL_OPTIONS[0]["id"])
                    system_prompt = st.session_state.get("custom_system_prompt", DEFAULT_SYSTEM_PROMPT)
                    temperature = st.session_state.get("temperature", 0.7)
                    
                    response = llm_service.call_llm(
                        messages=api_messages_formatted,
                        model=model,
                        system_prompt=system_prompt,
                        temperature=temperature
                    )
                    
                    assistant_response = response["choices"][0]["message"]["content"]
                    
                    # Aktualizuj statystyki tokenów
                    if "usage" in response:
                        usage = response["usage"]
                        st.session_state["token_usage"]["prompt"] += usage["prompt_tokens"]
                        st.session_state["token_usage"]["completion"] += usage["completion_tokens"]
                        
                        # Oblicz koszt
                        cost = calculate_cost(
                            model, 
                            usage["prompt_tokens"], 
                            usage["completion_tokens"]
                        )
                        
                        st.session_state["token_usage"]["cost"] += cost
                    
                    # Wyświetl odpowiedź asystenta
                    with st.chat_message("assistant"):
                        st.markdown(assistant_response)
                    
                    # Zapisz odpowiedź asystenta
                    db.save_message(current_conversation_id, "assistant", assistant_response)
                    
                    # Wyczyść załączniki po wysłaniu
                    st.session_state["attachments"] = []
                    
                except Exception as e:
                    st.error(f"Wystąpił błąd: {str(e)}")

@requires_auth
def knowledge_base_component():
    """Komponent bazy wiedzy"""
    st.title("📚 Baza Wiedzy")
    
    db = st.session_state.get("db")
    if not db:
        st.error("⚠️ Nie można zainicjalizować bazy danych. Odśwież stronę i spróbuj ponownie.")
        return
    
    tab1, tab2 = st.tabs(["Przeglądaj", "Dodaj nowy"])
    
    with tab1:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input("🔍 Szukaj w bazie wiedzy", placeholder="Wpisz zapytanie...")
        
        with col2:
            categories = ["Wszystkie"] + KNOWLEDGE_CATEGORIES
            selected_category = st.selectbox("Kategoria", categories)
        
        # Pobierz i wyświetl elementy bazy wiedzy
        if search_query:
            items = db.search_knowledge_base(search_query, selected_category if selected_category != "Wszystkie" else None)
        else:
            items = db.get_knowledge_items(selected_category if selected_category != "Wszystkie" else None)
        
        if not items:
            st.info("Nie znaleziono pasujących elementów bazy wiedzy.")
        else:
            for item in items:
                with st.expander(f"**{item['title']}** ({item['category']})"):
                    st.markdown(item["content"])
                    
                    col1, col2 = st.columns([1, 6])
                    with col1:
                        if st.button("Usuń", key=f"del_kb_{item['id']}", help="Usuń ten element z bazy wiedzy"):
                            db.delete_knowledge_item(item["id"])
                            st.success("Element usunięty!")
                            st.rerun()
                    
                    with col2:
                        if st.button("Edytuj", key=f"edit_kb_{item['id']}", help="Edytuj ten element"):
                            st.session_state["editing_item"] = item
                            st.rerun()
    
    with tab2:
        # Edycja istniejącego elementu
        editing_item = st.session_state.get("editing_item")
        
        if editing_item:
            st.subheader("Edytuj element bazy wiedzy")
            item_title = st.text_input("Tytuł", value=editing_item["title"])
            item_category = st.selectbox("Kategoria", KNOWLEDGE_CATEGORIES, index=KNOWLEDGE_CATEGORIES.index(editing_item["category"]) if editing_item["category"] in KNOWLEDGE_CATEGORIES else 0)
            item_content = st.text_area("Zawartość (wspierane Markdown)", value=editing_item["content"], height=300)
            item_tags = st.multiselect("Tagi (opcjonalnie)", options=["Kod", "Komponent", "Integracja", "Design"], default=editing_item["tags"])
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Anuluj edycję"):
                    st.session_state.pop("editing_item", None)
                    st.rerun()
            
            with col2:
                if st.button("Zapisz zmiany"):
                    db.update_knowledge_item(
                        editing_item["id"],
                        item_title,
                        item_content,
                        item_category,
                        item_tags
                    )
                    st.success("Element zaktualizowany!")
                    st.session_state.pop("editing_item", None)
                    st.rerun()
        
        # Dodawanie nowego elementu
        else:
            st.subheader("Dodaj nowy element do bazy wiedzy")
            item_title = st.text_input("Tytuł", placeholder="Np. Komponent wyboru plików z podglądem")
            item_category = st.selectbox("Kategoria", KNOWLEDGE_CATEGORIES)
            item_content = st.text_area("Zawartość (wspierane Markdown)", placeholder="Wpisz zawartość, kod, fragmenty...", height=300)
            item_tags = st.multiselect("Tagi (opcjonalnie)", options=["Kod", "Komponent", "Integracja", "Design"])
            
            if st.button("Dodaj do bazy wiedzy"):
                if item_title and item_content:
                    db.save_knowledge_item(
                        item_title,
                        item_content,
                        item_category,
                        item_tags
                    )
                    st.success("Dodano do bazy wiedzy!")
                    # Wyczyść pola po dodaniu
                    st.session_state["new_item_title"] = ""
                    st.session_state["new_item_content"] = ""
                    st.rerun()
                else:
                    st.error("Tytuł i zawartość są wymagane.")

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
    page = sidebar_component()
    
    # Wyświetl główny komponent
    if page == "Chat Asystent":
        chat_component()
    elif page == "Baza Wiedzy":
        knowledge_base_component()

if __name__ == "__main__":
    main()
