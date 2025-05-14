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
import base64
import io

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
        "name": "Claude 3.7 Thinking",
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
                model: str = "anthropic/claude-3.7-sonnet:floor", 
                system_prompt: str = None, 
                temperature: float = 0.7, 
                max_tokens: int = 8000,  # Zwiększenie max_tokens dla dłuższych odpowiedzi
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
                    timeout=60  # Zwiększenie timeout dla dłuższych odpowiedzi
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

# === Personalizacja interfejsu ===
def apply_custom_css():
    """Dodaj niestandardowy CSS do aplikacji"""
    st.markdown("""
    <style>
    /* Styl dla kontenera wiadomości - wypełnia całą stronę */
    .main .block-container {
        padding-bottom: 5rem;  /* Miejsce na input na dole */
    }
    
    /* Przytwierdzony pasek wprowadzania na dole ekranu */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0; 
        right: 0;
        background-color: white;
        padding: 1rem;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        z-index: 1000;
    }
    
    /* Stylowanie załączników */
    .attachment-button {
        padding: 0.2rem 0.5rem;
        font-size: 0.8rem;
        border-radius: 999px;
        background-color: #f0f2f6;
        border: 1px solid #dfe1e5;
        margin-right: 0.3rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 150px;
    }
    
    /* Większa odległość od dołu ekranu przy wiadomościach */
    .stChatMessage {
        margin-bottom: 0.5rem;
    }
    
    /* Przepełnienie dla długich bloków kodu */
    .stChatMessage code {
        white-space: pre-wrap;
        word-break: break-word;
    }
    
    .stChatMessage pre {
        overflow-x: auto;
        max-width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

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
            st.rerun()

@requires_auth
def chat_component():
    """Komponent interfejsu czatu"""
    # Dodaj niestandardowy CSS
    apply_custom_css()
    
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

    if "attached_images" not in st.session_state:
        st.session_state["attached_images"] = {}
        
    # Tworzymy placeholder dla wiadomości - będą wyświetlane tu
    messages_container = st.container()
    
    # Pobierz wiadomości z DB
    messages = db.get_messages(current_conversation_id)
    
    # Obsługa nowej wiadomości użytkownika
    # Najpierw umieszczamy kontener formularza wejściowego (będzie na dole)
    st.markdown('<div class="input-container" id="input-container">', unsafe_allow_html=True)
        
    # Input użytkownika i przyciski załączników
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.chat_input("Wpisz swoje pytanie lub zadanie...")

    with col2:
        # Przyciski załączników w małej kolumnie obok pola czatu
        attachment_col1, attachment_col2, attachment_col3 = st.columns(3)
        with attachment_col1:
            if st.button("📷", help="Dodaj obraz", key="btn_img"):
                st.session_state["show_image_uploader"] = not st.session_state.get("show_image_uploader", False)
                st.rerun()
                
        with attachment_col2:
            if st.button("📄", help="Dodaj plik", key="btn_file"):
                st.session_state["show_file_uploader"] = not st.session_state.get("show_file_uploader", False)
                st.rerun()
                
        with attachment_col3:
            if st.button("💻", help="Dodaj kod", key="btn_code"):
                st.session_state["show_code_input"] = not st.session_state.get("show_code_input", False)
                st.rerun()

    # Wyświetl liczbę załączników jako małe etykiety
    if st.session_state["attachments"]:
        st.caption(f"Załączników: {len(st.session_state['attachments'])}")
        # Przyciski załączników jako inline elementy
        attachment_html = ""
        for i, attachment in enumerate(st.session_state["attachments"]):
            attachment_name = attachment.get('name', 'Załącznik')[:10] + (attachment.get('name', 'Załącznik')[10:] and '...')
            attachment_html += f"""
            <button class="attachment-button" onclick="removeAttachment({i})" title="Kliknij, aby usunąć">
                {"📷" if attachment.get("type") == "image" else "📄"} {attachment_name}
            </button>
            """
        
        st.markdown(f"""
        <div>{attachment_html}</div>
        <script>
        function removeAttachment(index) {{
            // Symuluj kliknięcie odpowiedniego niewidocznego przycisku
            document.getElementById('remove_' + index).click();
        }}
        </script>
        """, unsafe_allow_html=True)
        
        # Niewidoczne przyciski do usuwania załączników
        for i, attachment in enumerate(st.session_state["attachments"]):
            if st.button("Usuń", key=f"remove_{i}", help="Usuń załącznik", visible=False):
                if attachment.get("type") == "image" and attachment.get("name") in st.session_state["attached_images"]:
                    del st.session_state["attached_images"][attachment.get("name")]
                st.session_state["attachments"].pop(i)
                st.rerun()

    # Formularze załączników - pokazywane tylko gdy użytkownik kliknie odpowiedni przycisk
    if st.session_state.get("show_image_uploader", False):
        with st.expander("Dodaj obraz", expanded=True):
            uploaded_file = st.file_uploader("Wybierz obraz", type=["png", "jpg", "jpeg"], key="image_upload")
            if uploaded_file is not None and st.button("Dodaj"):
                # Zapisujemy obraz w pamięci sesji
                image_name = uploaded_file.name
                st.session_state["attached_images"][image_name] = uploaded_file.getvalue()
                
                # Do załączników dodajemy tylko referencję
                st.session_state["attachments"].append({
                    "type": "image",
                    "name": image_name
                })
                st.session_state["show_image_uploader"] = False
                st.success(f"Dodano obraz: {image_name}")
                st.rerun()

    if st.session_state.get("show_file_uploader", False):
        with st.expander("Dodaj plik tekstowy", expanded=True):
            uploaded_file = st.file_uploader("Wybierz plik", type=["txt", "md", "json", "csv"], key="text_upload")
            if uploaded_file is not None and st.button("Dodaj"):
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
            code_language = st.selectbox("Język programowania", ["python", "javascript", "html", "css", "json", "sql", "bash"], key="code_language")
            code_content = st.text_area("Wklej kod", height=150, key="code_content")
            file_name = st.text_input("Nazwa pliku (opcjonalnie)", value=f"code.{code_language}", key="code_filename")

            if st.button("Dodaj") and code_content:
                st.session_state["attachments"].append({
                    "type": "file",
                    "name": file_name,
                    "text_content": f"```{code_language}\n{code_content}\n```"
                })
                st.session_state["show_code_input"] = False
                st.success(f"Dodano kod: {file_name}")
                st.rerun()
                
    # Zamknięcie kontenera wejściowego
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Teraz wyświetlamy wiadomości we wcześniej utworzonym kontenerze
    with messages_container:
        # Wyświetl istniejące wiadomości
        for message in messages:
            role = message["role"]
            content = format_message_for_display(message)

            if role == "user":
                with st.chat_message("user"):
                    st.markdown(content)
                    # Wyświetl załączniki jeśli istnieją - tylko obrazy, bez wyświetlania tekstu załączników
                    for attachment in message.get("attachments", []):
                        if attachment.get("type") == "image":
                            try:
                                # Załączniki obrazów są trzymane w sesji, a nie w DB
                                if "attached_images" in st.session_state and attachment.get("name") in st.session_state["attached_images"]:
                                    img_data = st.session_state["attached_images"][attachment.get("name")]
                                    st.image(img_data, caption=attachment.get("name", "Załącznik"))
                            except Exception as e:
                                st.error(f"Nie można wyświetlić obrazu: {str(e)}")

            elif role == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(content)

    # Obsługa wprowadzonej wiadomości - po user_input, ale przed końcem funkcji
    if user_input:
        # Natychmiast wyświetl wiadomość użytkownika
        with messages_container:
            with st.chat_message("user"):
                st.markdown(user_input)
                # Pokaż załączniki
                for attachment in st.session_state.get("attachments", []):
                    if attachment.get("type") == "image":
                        try:
                            if "attached_images" in st.session_state and attachment.get("name") in st.session_state["attached_images"]:
                                img_data = st.session_state["attached_images"][attachment.get("name")]
                                st.image(img_data, caption=attachment.get("name", "Załącznik"))
                        except Exception as e:
                            st.error(f"Nie można wyświetlić obrazu: {str(e)}")
        
        # Przygotuj treść wiadomości i załączniki
        message_content = user_input

        # Kopiujemy załączniki ze stanu sesji
        attachments_to_send = []
        for attachment in st.session_state.get("attachments", []):
            attachment_copy = attachment.copy()
            attachments_to_send.append(attachment_copy)

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

        # Zapisz wiadomość użytkownika w bazie danych
        db.save_message(current_conversation_id, "user", user_input, attachments_to_send)

        # Przygotuj wiadomości dla API
        api_messages = []
        for msg in messages:
            api_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # Dodaj aktualną wiadomość użytkownika
        api_messages.append({"role": "user", "content": message_content})

        # Pokaż spinner podczas oczekiwania na odpowiedź
        with messages_container:
            with st.chat_message("assistant"):
                with st.spinner("Generowanie odpowiedzi..."):
                    try:
                        model = st.session_state.get("model_selection", MODEL_OPTIONS[0]["id"])
                        system_prompt = st.session_state.get("custom_system_prompt", DEFAULT_SYSTEM_PROMPT)
                        temperature = st.session_state.get("temperature", 0.7)

                        response = llm_service.call_llm(
                            messages=api_messages,
                            model=model,
                            system_prompt=system_prompt,
                            temperature=temperature,
                            max_tokens=8000
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

                        # Zapisz odpowiedź asystenta
                        db.save_message(current_conversation_id, "assistant", assistant_response)
                        
                        # Wyświetl odpowiedź
                        st.markdown(assistant_response)
                        
                        # Wyczyść załączniki po wysłaniu
                        st.session_state["attachments"] = []
                        # Ukryj formularze załączników
                        st.session_state["show_image_uploader"] = False
                        st.session_state["show_file_uploader"] = False  
                        st.session_state["show_code_input"] = False
                        
                    except Exception as e:
                        st.error(f"Wystąpił błąd: {str(e)}")
                        st.error("Szczegóły: " + str(type(e)))

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
