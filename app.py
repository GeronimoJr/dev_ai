import streamlit as st
import sqlite3
import datetime
import os
import time
import base64
import io
import json
import random
import string
import requests
from PIL import Image
import openai
import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple, Union
import logging

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Konfiguracja strony Streamlit
st.set_page_config(
    page_title="FloorDev AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Klasa do obsługi bazy danych
class AssistantDB:
    def __init__(self, db_path="assistant_data.db"):
        """Inicjalizuje połączenie z bazą danych."""
        self.db_path = db_path
        self.init_db()

    def get_connection(self):
        """Tworzy i zwraca połączenie z bazą danych."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Inicjalizuje bazę danych, tworząc tabele jeśli nie istnieją."""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Tworzenie tabeli użytkowników
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Tworzenie tabeli konwersacji
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')

        # Tworzenie tabeli wiadomości
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            role TEXT,
            content TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            attachments TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations (id)
        )
        ''')

        conn.commit()
        conn.close()

    def create_user(self, username, password):
        """Tworzy nowego użytkownika w bazie danych."""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Haszowanie hasła
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        try:
            cursor.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, password_hash)
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Użytkownik już istnieje
            return False
        finally:
            conn.close()

    def verify_user(self, username, password):
        """Weryfikuje dane logowania użytkownika."""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Haszowanie hasła
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        cursor.execute(
            "SELECT id FROM users WHERE username = ? AND password_hash = ?",
            (username, password_hash)
        )
        user = cursor.fetchone()
        conn.close()

        return user['id'] if user else None

    def get_user_id(self, username):
        """Pobiera ID użytkownika na podstawie nazwy użytkownika."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()

        return user['id'] if user else None

    def create_conversation(self, user_id, title=None):
        """Tworzy nową konwersację dla użytkownika."""
        if not title:
            title = f"Nowa konwersacja {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"

        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO conversations (user_id, title) VALUES (?, ?)",
            (user_id, title)
        )
        conversation_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return conversation_id

    def update_conversation_title(self, conversation_id, new_title):
        """Aktualizuje tytuł konwersacji."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE conversations SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (new_title, conversation_id)
        )
        conn.commit()
        conn.close()

    def get_conversations(self, user_id):
        """Pobiera wszystkie konwersacje użytkownika."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, title, created_at FROM conversations WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,)
        )
        conversations = cursor.fetchall()
        conn.close()

        return [dict(conv) for conv in conversations]

    def delete_conversation(self, conversation_id):
        """Usuwa konwersację i jej wiadomości."""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Najpierw usuwamy wiadomości
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))

        # Następnie usuwamy konwersację
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

        conn.commit()
        conn.close()

    def add_message(self, conversation_id, role, content, attachments=None):
        """Dodaje nową wiadomość do konwersacji."""
        conn = self.get_connection()
        cursor = conn.cursor()

        attachments_json = json.dumps(attachments) if attachments else None

        cursor.execute(
            "INSERT INTO messages (conversation_id, role, content, attachments) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, attachments_json)
        )

        # Aktualizacja czasu ostatniej modyfikacji konwersacji
        cursor.execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (conversation_id,)
        )

        conn.commit()
        conn.close()

    def get_messages(self, conversation_id):
        """Pobiera wszystkie wiadomości z konwersacji."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, role, content, timestamp, attachments FROM messages WHERE conversation_id = ? ORDER BY timestamp",
            (conversation_id,)
        )
        messages = cursor.fetchall()
        conn.close()

        result = []
        for msg in messages:
            message_dict = dict(msg)
            if message_dict['attachments']:
                message_dict['attachments'] = json.loads(message_dict['attachments'])
            else:
                message_dict['attachments'] = None
            result.append(message_dict)

        return result

# Klasa do obsługi modelu językowego
class LLMService:
    def __init__(self, api_key=None, model="gpt-4-turbo"):
        """Inicjalizuje usługę modelu językowego."""
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model

        # Ustawienie klucza API
        if self.api_key:
            openai.api_key = self.api_key
        else:
            logging.warning("Brak klucza API dla modelu językowego!")

    def generate_response(self, messages, max_retries=3, retry_delay=2):
        """Generuje odpowiedź modelu językowego z obsługą ponownych prób."""
        if not self.api_key:
            return "Brak klucza API dla modelu językowego. Skonfiguruj klucz w ustawieniach."

        # Przygotowanie wiadomości dla API
        formatted_messages = []
        for msg in messages:
            message = {"role": msg["role"], "content": []}

            # Dodanie głównej treści jako tekstu
            message["content"].append({"type": "text", "text": msg["content"]})

            # Dodanie załączników, jeśli istnieją
            if msg.get("attachments"):
                for attachment in msg["attachments"]:
                    if attachment["type"] == "image":
                        # Dodanie obrazu jako załącznika
                        message["content"].append({
                            "type": "image_url",
                            "image_url": {
                                "url": attachment["data"],
                                "detail": "high"
                            }
                        })

            formatted_messages.append(message)

        for attempt in range(max_retries):
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=formatted_messages,
                    temperature=0.7,
                    max_tokens=4000
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    logging.warning(f"Próba {attempt+1} nie powiodła się: {str(e)}. Ponowna próba za {retry_delay} s...")
                    time.sleep(retry_delay)
                else:
                    logging.error(f"Wszystkie próby nie powiodły się: {str(e)}")
                    return f"Wystąpił błąd podczas komunikacji z modelem: {str(e)}"

# Inicjalizacja bazy danych i usługi LLM
db = AssistantDB()
llm_service = LLMService()

# Funkcje pomocnicze
def process_code_blocks(text):
    """Przetwarza bloki kodu w tekście Markdown."""
    # Wzorzec do wykrywania bloków kodu
    pattern = r'```(\w+)?\n(.*?)\n```'

    # Funkcja do przetwarzania znalezionych bloków
    def process_match(match):
        language = match.group(1) or ''
        code = match.group(2)
        return f'```{language}\n{code}\n```'

    # Przetwarzanie tekstu z flagą re.DOTALL, aby dopasować wiele linii
    processed_text = re.sub(pattern, process_match, text, flags=re.DOTALL)
    return processed_text

def encode_image(image_file):
    """Koduje obraz do formatu base64 dla API."""
    return f"data:image/jpeg;base64,{base64.b64encode(image_file.getvalue()).decode('utf-8')}"

def read_text_file(file):
    """Odczytuje zawartość pliku tekstowego."""
    try:
        content = file.getvalue().decode('utf-8')
        return content
    except UnicodeDecodeError:
        return "Nie można odczytać pliku - to nie jest plik tekstowy."

# Inicjalizacja stanu sesji
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "user_id" not in st.session_state:
    st.session_state["user_id"] = None

if "username" not in st.session_state:
    st.session_state["username"] = None

if "current_conversation_id" not in st.session_state:
    st.session_state["current_conversation_id"] = None

if "show_image_uploader" not in st.session_state:
    st.session_state["show_image_uploader"] = False

if "show_file_uploader" not in st.session_state:
    st.session_state["show_file_uploader"] = False

if "show_code_input" not in st.session_state:
    st.session_state["show_code_input"] = False

# Dodaj niestandardowy CSS dla przypiętego paska wejściowego i przycisku załącznika
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
    display: flex;
    align-items: center;
}

/* Stylowanie przycisku załącznika */
.attachment-button {
    margin-left: 10px;
    background-color: #f0f2f6;
    border-radius: 50%;
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    border: 1px solid #ddd;
}

.attachment-button:hover {
    background-color: #e0e2e6;
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

<script>
// Dodajemy przycisk załącznika obok pola wejściowego
document.addEventListener('DOMContentLoaded', function() {
    const chatInputContainer = document.querySelector('.stChatInputContainer');
    if (chatInputContainer) {
        const attachButton = document.createElement('div');
        attachButton.className = 'attachment-button';
        attachButton.innerHTML = '📎';
        attachButton.title = 'Dodaj załącznik';
        attachButton.onclick = function() {
            // Wywołanie funkcji Streamlit poprzez kliknięcie ukrytego przycisku
            document.getElementById('attach_button_trigger').click();
        };
        chatInputContainer.appendChild(attachButton);
    }
});
</script>
""", unsafe_allow_html=True)

# Funkcja do logowania
def login_page():
    st.title("FloorDev AI Assistant")

    tab1, tab2 = st.tabs(["Logowanie", "Rejestracja"])

    with tab1:
        username = st.text_input("Nazwa użytkownika", key="login_username")
        password = st.text_input("Hasło", type="password", key="login_password")

        if st.button("Zaloguj się", key="login_button"):
            user_id = db.verify_user(username, password)
            if user_id:
                st.session_state["logged_in"] = True
                st.session_state["user_id"] = user_id
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Nieprawidłowa nazwa użytkownika lub hasło.")

    with tab2:
        new_username = st.text_input("Nazwa użytkownika", key="register_username")
        new_password = st.text_input("Hasło", type="password", key="register_password")
        confirm_password = st.text_input("Potwierdź hasło", type="password", key="confirm_password")

        if st.button("Zarejestruj się", key="register_button"):
            if new_password != confirm_password:
                st.error("Hasła nie są zgodne.")
            elif len(new_password) < 4:
                st.error("Hasło musi mieć co najmniej 4 znaki.")
            else:
                success = db.create_user(new_username, new_password)
                if success:
                    st.success("Rejestracja udana. Możesz się teraz zalogować.")
                else:
                    st.error("Nazwa użytkownika jest już zajęta.")

# Funkcja do wyświetlania konwersacji
def main_page():
    # Sidebar z listą konwersacji
    with st.sidebar:
        st.title(f"Witaj, {st.session_state['username']}!")

        if st.button("Nowa konwersacja", key="new_conversation"):
            conversation_id = db.create_conversation(st.session_state["user_id"])
            st.session_state["current_conversation_id"] = conversation_id
            st.rerun()

        st.subheader("Twoje konwersacje")

        conversations = db.get_conversations(st.session_state["user_id"])
        for conv in conversations:
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(conv["title"], key=f"conv_{conv['id']}", use_container_width=True):
                    st.session_state["current_conversation_id"] = conv["id"]
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"delete_{conv['id']}"):
                    db.delete_conversation(conv["id"])
                    if st.session_state["current_conversation_id"] == conv["id"]:
                        st.session_state["current_conversation_id"] = None
                    st.rerun()

        if st.button("Wyloguj się", key="logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Główny obszar konwersacji
    if st.session_state["current_conversation_id"]:
        conversation_id = st.session_state["current_conversation_id"]

        # Pobierz tytuł konwersacji
        conversations = db.get_conversations(st.session_state["user_id"])
        current_conversation = next((c for c in conversations if c["id"] == conversation_id), None)

        if current_conversation:
            # Edytowalny tytuł konwersacji
            new_title = st.text_input("Tytuł konwersacji", value=current_conversation["title"], key="conversation_title")
            if new_title != current_conversation["title"]:
                db.update_conversation_title(conversation_id, new_title)
                st.rerun()

        # Wyświetlanie wiadomości
        messages = db.get_messages(conversation_id)

        # Kontener na wiadomości
        message_container = st.container()

        with message_container:
            for msg in messages:
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.write(msg["content"])

                        # Wyświetlanie załączników
                        if msg["attachments"]:
                            for attachment in msg["attachments"]:
                                if attachment["type"] == "image":
                                    st.image(attachment["data"])
                                elif attachment["type"] == "text":
                                    with st.expander("Załącznik tekstowy"):
                                        st.text(attachment["data"])
                                elif attachment["type"] == "code":
                                    with st.expander(f"Kod ({attachment['language']})"):
                                        st.code(attachment["data"], language=attachment["language"])

                elif msg["role"] == "assistant":
                    with st.chat_message("assistant"):
                        # Przetwarzanie bloków kodu w odpowiedzi
                        processed_content = process_code_blocks(msg["content"])
                        st.markdown(processed_content)

        # Ukryty przycisk do obsługi kliknięcia przycisku załącznika
        if "show_attachment_menu" not in st.session_state:
            st.session_state["show_attachment_menu"] = False

        # Ukryty przycisk, który będzie kliknięty przez JavaScript
        button_clicked = st.button('📎', key='attach_button_trigger', help="Dodaj załącznik", style="display: none;")
        if button_clicked:
            st.session_state["show_attachment_menu"] = not st.session_state["show_attachment_menu"]
            st.rerun()

        # Pole wejściowe użytkownika
        user_input = st.chat_input("Wpisz swoje pytanie lub zadanie...")

        # Wyświetlanie menu załączników, jeśli zostało aktywowane
        if st.session_state["show_attachment_menu"]:
            with st.container():
                st.markdown("<div style='background-color: white; padding: 10px; border-radius: 5px; border: 1px solid #ddd; margin-bottom: 10px;'>", unsafe_allow_html=True)
                st.write("Wybierz typ załącznika:")

                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

                with col1:
                    if st.button("📷 Obraz", use_container_width=True):
                        st.session_state["show_image_uploader"] = True
                        st.session_state["show_file_uploader"] = False
                        st.session_state["show_code_input"] = False
                        st.session_state["show_attachment_menu"] = False
                        st.rerun()

                with col2:
                    if st.button("📄 Plik", use_container_width=True):
                        st.session_state["show_image_uploader"] = False
                        st.session_state["show_file_uploader"] = True
                        st.session_state["show_code_input"] = False
                        st.session_state["show_attachment_menu"] = False
                        st.rerun()

                with col3:
                    if st.button("💻 Kod", use_container_width=True):
                        st.session_state["show_image_uploader"] = False
                        st.session_state["show_file_uploader"] = False
                        st.session_state["show_code_input"] = True
                        st.session_state["show_attachment_menu"] = False
                        st.rerun()

                with col4:
                    if st.button("❌ Anuluj", use_container_width=True):
                        st.session_state["show_attachment_menu"] = False
                        st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)

        # Obsługa przesyłania obrazu
        if st.session_state["show_image_uploader"]:
            uploaded_image = st.file_uploader("Wybierz obraz", type=["png", "jpg", "jpeg"], key="image_uploader")

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Dodaj obraz", key="add_image"):
                    if uploaded_image:
                        # Kodowanie obrazu do base64
                        image_data = encode_image(uploaded_image)

                        # Dodanie wiadomości z załącznikiem obrazu
                        db.add_message(
                            conversation_id, 
                            "user", 
                            "Załączony obraz:",
                            [{"type": "image", "data": image_data}]
                        )

                        # Generowanie odpowiedzi asystenta
                        messages = db.get_messages(conversation_id)
                        formatted_messages = [{"role": msg["role"], "content": msg["content"], "attachments": msg["attachments"]} for msg in messages]

                        with st.spinner("Generowanie odpowiedzi..."):
                            assistant_response = llm_service.generate_response(formatted_messages)
                            db.add_message(conversation_id, "assistant", assistant_response)

                        # Resetowanie stanu
                        st.session_state["show_image_uploader"] = False
                        st.rerun()

            with col2:
                if st.button("Anuluj", key="cancel_image"):
                    st.session_state["show_image_uploader"] = False
                    st.rerun()

        # Obsługa przesyłania pliku
        if st.session_state["show_file_uploader"]:
            uploaded_file = st.file_uploader("Wybierz plik tekstowy", type=["txt", "md", "py", "js", "html", "css", "json", "csv"], key="file_uploader")

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Dodaj plik", key="add_file"):
                    if uploaded_file:
                        # Odczytanie zawartości pliku
                        file_content = read_text_file(uploaded_file)

                        # Dodanie wiadomości z załącznikiem tekstowym
                        db.add_message(
                            conversation_id, 
                            "user", 
                            f"Załączony plik: {uploaded_file.name}",
                            [{"type": "text", "data": file_content}]
                        )

                        # Generowanie odpowiedzi asystenta
                        messages = db.get_messages(conversation_id)
                        formatted_messages = [{"role": msg["role"], "content": msg["content"], "attachments": msg["attachments"]} for msg in messages]

                        with st.spinner("Generowanie odpowiedzi..."):
                            assistant_response = llm_service.generate_response(formatted_messages)
                            db.add_message(conversation_id, "assistant", assistant_response)

                        # Resetowanie stanu
                        st.session_state["show_file_uploader"] = False
                        st.rerun()

            with col2:
                if st.button("Anuluj", key="cancel_file"):
                    st.session_state["show_file_uploader"] = False
                    st.rerun()

        # Obsługa wprowadzania kodu
        if st.session_state["show_code_input"]:
            language = st.selectbox("Język programowania", ["python", "javascript", "html", "css", "sql", "bash", "json", "plaintext"])
            code_content = st.text_area("Wklej swój kod", height=200)

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Dodaj kod", key="add_code"):
                    if code_content:
                        # Dodanie wiadomości z załącznikiem kodu
                        db.add_message(
                            conversation_id, 
                            "user", 
                            f"Załączony kod ({language}):",
                            [{"type": "code", "language": language, "data": code_content}]
                        )

                        # Generowanie odpowiedzi asystenta
                        messages = db.get_messages(conversation_id)
                        formatted_messages = [{"role": msg["role"], "content": msg["content"], "attachments": msg["attachments"]} for msg in messages]

                        with st.spinner("Generowanie odpowiedzi..."):
                            assistant_response = llm_service.generate_response(formatted_messages)
                            db.add_message(conversation_id, "assistant", assistant_response)

                        # Resetowanie stanu
                        st.session_state["show_code_input"] = False
                        st.rerun()

            with col2:
                if st.button("Anuluj", key="cancel_code"):
                    st.session_state["show_code_input"] = False
                    st.rerun()

        # Obsługa wysyłania wiadomości tekstowej
        if user_input:
            # Dodanie wiadomości użytkownika
            db.add_message(conversation_id, "user", user_input)

            # Generowanie odpowiedzi asystenta
            messages = db.get_messages(conversation_id)
            formatted_messages = [{"role": msg["role"], "content": msg["content"], "attachments": msg["attachments"]} for msg in messages]

            with st.spinner("Generowanie odpowiedzi..."):
                assistant_response = llm_service.generate_response(formatted_messages)
                db.add_message(conversation_id, "assistant", assistant_response)

            st.rerun()

    else:
        st.title("FloorDev AI Assistant")
        st.write("Wybierz konwersację z listy lub utwórz nową, aby rozpocząć.")

# Główna funkcja aplikacji
def main():
    if not st.session_state["logged_in"]:
        login_page()
    else:
        main_page()

if __name__ == "__main__":
    main()
