import tkinter as tk
from tkinter import messagebox, filedialog, Toplevel, Frame, Scrollbar, Text, RIGHT, Y, END
import itertools
from LLM import initialize_client, send_query
from docread import extract_text_from_pdf
import json
import os
from raggy_hand import RAGSystem

# Global variables
api_key = None
client = None
history = []
text_storage = ""
selected_model = "llama-3.3-70b-versatile"  # Default model
chat_concatenation = False  # Chat concatenation toggle state


# Function to handle API key input
def set_api_key():
    global api_key, client
    api_key = api_key_entry.get()
    if not api_key.strip():
        messagebox.showerror("API Key Missing", "Please provide a valid API key.")
        return
    try:
        client = initialize_client(api_key)
        messagebox.showinfo("Success", "API key is valid!")
    except ValueError as e:
        messagebox.showerror("Error", str(e))
        return
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        return

    # Switch frames
    api_key_frame.pack_forget()
    main_frame.pack()


# Function to toggle query entry size
def toggle_query_entry():
    if query_entry["height"] == 5:
        query_entry.config(height=15)
        toggle_button.config(text="Collapse Query Input")
    else:
        query_entry.config(height=5)
        toggle_button.config(text="Expand Query Input")


# Function to update selected model
def update_selected_model(*args):
    global selected_model
    selected_model = llm_model_var.get()
    print(f"Model changed to: {selected_model}")


# Function to toggle chat concatenation
def toggle_chat_concatenation():
    global chat_concatenation
    chat_concatenation = not chat_concatenation
    status = "ON" if chat_concatenation else "OFF"
    concat_toggle_button.config(text=f"Chat Context: {status}")
    concat_status_label.config(
        text=f"Context Mode: {'Enabled' if chat_concatenation else 'Disabled'}",
        fg="#27AE60" if chat_concatenation else "#E74C3C"
    )


# Function to handle sending user input to the API
def get_response():
    global history, selected_model, chat_concatenation
    if not client:
        messagebox.showerror("API Key Missing", "API key is required to make queries.")
        return

    user_input = query_entry.get("1.0", tk.END).strip()
    if not user_input.strip():
        messagebox.showwarning("Input Required", "Please enter a query.")
        return

    try:
        # Build messages based on chat concatenation setting
        if chat_concatenation and history:
            messages = []
            for prev_query, prev_response in history:
                messages.append({"role": "user", "content": prev_query})
                messages.append({"role": "assistant", "content": prev_response})
            messages.append({"role": "user", "content": user_input})
        else:
            messages = [{"role": "user", "content": user_input}]

        response = send_query(client, user_input, model=selected_model, stream=True,
                              messages=messages if chat_concatenation else None)
        response_text = ""

        query_display.config(state=tk.NORMAL)
        query_display.delete("1.0", tk.END)
        query_display.insert(tk.END, "Loading...\n")
        query_display.update_idletasks()

        for chunk in response:
            if chunk.strip():
                if chunk.startswith("data: "):
                    chunk = chunk[len("data: "):]
                if chunk == "[DONE]":
                    break
                try:
                    data = json.loads(chunk)
                    content = data["choices"][0]["delta"].get("content", "")
                    if content:
                        response_text += content
                        query_display.delete("1.0", tk.END)
                        query_display.insert(tk.END, response_text)
                        query_display.update_idletasks()
                except json.JSONDecodeError:
                    continue

        history.append((user_input, response_text))
        query_display.config(state=tk.DISABLED)

    except Exception as e:
        query_display.config(state=tk.NORMAL)
        query_display.delete("1.0", tk.END)
        query_display.insert(tk.END, f"Error: {str(e)}")
        query_display.config(state=tk.DISABLED)
        messagebox.showerror("Error", f"An error occurred: {e}")


# Function to clear the query text widget
def clear_text():
    query_display.config(state=tk.NORMAL)
    query_display.delete("1.0", tk.END)
    query_display.config(state=tk.DISABLED)


# Function to view query history
def view_history():
    history_window = Toplevel(main)
    history_window.title("History")
    history_window.geometry("600x400")

    history_text = Text(history_window, wrap=tk.WORD, state=tk.NORMAL, bg="#E4E2E2", fg="#474043")
    history_text.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    if not history:
        history_text.insert(tk.END, "No history available.")
    else:
        for i, (query, response) in enumerate(history, start=1):
            history_text.insert(tk.END, f"Query {i}: {query}\n\nResponse: {response}\n\n{'-' * 60}\n\n")

    history_text.config(state=tk.DISABLED)


# Function to show current response
def show_current_response():
    if not history:
        messagebox.showwarning("No Response Available", "No query has been made yet.")
        return

    last_query, last_response = history[-1]
    current_window = Toplevel(main)
    current_window.title("Current Response")
    current_window.geometry("600x400")

    current_text = Text(current_window, wrap=tk.WORD, bg="#FFFFFF", fg="#000000")
    current_text.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
    current_text.insert(tk.END, f"Query:\n{last_query}\n\nResponse:\n{last_response}")
    current_text.config(state=tk.DISABLED)


# File handling functions
def upload_file():
    global text_storage
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if file_path:
        extracted_text = extract_text_from_pdf(file_path)
        if extracted_text.startswith("Error"):
            messagebox.showerror("Error", extracted_text)
        else:
            text_storage = extracted_text
            messagebox.showinfo("Success", "PDF text extracted successfully!")
    else:
        messagebox.showinfo("No File Selected", "Please select a PDF file to upload.")


def view_text_window():
    if not text_storage.strip():
        messagebox.showwarning("No Text Available", "No text has been extracted yet. Upload a PDF first.")
        return

    new_window = Toplevel()
    new_window.title("Extracted Text")
    new_window.geometry("800x600")
    new_window.configure(bg="#8a9091")

    text_frame = Frame(new_window, bg="#68696b")
    text_frame.pack(pady=10, fill="both", expand=True)

    scrollbar = Scrollbar(text_frame)
    scrollbar.pack(side=RIGHT, fill=Y)

    text_display_new = Text(text_frame, wrap="word", font=("Arial", 12), bg="#68696b", yscrollcommand=scrollbar.set)
    text_display_new.pack(fill="both", expand=True)
    scrollbar.config(command=text_display_new.yview)
    text_display_new.insert(END, text_storage)


def save_to_file():
    global text_storage
    if text_storage.strip():
        save_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if save_path:
            try:
                with open(save_path, "w", encoding="utf-8") as file:
                    file.write(text_storage)
                messagebox.showinfo("Success", "Text saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {e}")
    else:
        messagebox.showwarning("No Text Available", "No text has been extracted yet. Upload a PDF first.")


def append_extracted_text():
    global text_storage
    if not text_storage.strip():
        messagebox.showwarning("No Text Available", "No text has been extracted yet. Upload a PDF first.")
        return

    current_query = query_entry.get("1.0", tk.END).strip()
    query_entry.delete("1.0", tk.END)
    query_entry.insert("1.0", f"{current_query}\n\n{text_storage}".strip())


# Animation function
def animation_text(frame):
    global animated_label, message_cycle, current_message

    animated_label = tk.Label(frame, text="", font=("Courier", 12), bg="#2C3E50", fg="#c29d9d")
    animated_label.pack(side=tk.BOTTOM, anchor='w', padx=10, pady=10)

    message = "Welcome to ShadowPine AI - Your Intelligent Assistant. Have fun exploring!"
    message_cycle = itertools.cycle(
        [message[:i] for i in range(1, len(message) + 1)] + [message[i:] for i in range(len(message))]
    )
    current_message = ""

    def animate_message():
        global current_message
        current_message = next(message_cycle)
        animated_label.config(text=current_message)
        animated_label.after(200, animate_message)

    animate_message()


# GUI setup
main = tk.Tk()
main.config(bg="#000000")
main.title("ShadowPine AI")
main.geometry("900x1000")

label = tk.Label(
    main,
    text="When unusual characters appear, it's important to find the\n"
         "source of the data and ask what encoding they used.\n"
         "In general, it's impossible to uncover the encoding\n"
         "given a sample file.\n"
         "There are a large number of valid byte code mappings\n"
         "that overlap between ASCII,",
    bg="#000000",
    fg="#FFFFFF",
    font=("Century", 12, "bold"),
    justify='left',
    anchor="w",
)
label.place(relx=0, rely=0.5, anchor="w")

file_label = tk.Label(main, text="ShadowPine", bg="#000000", fg="#FF0000", font=('Juice ITC', 36))
file_label.pack(pady=10)


# RAG Interface Class

class RAGInterface:
    def __init__(self, root, main_frame, api_key_frame):
        self.root = root
        self.main_frame = main_frame
        self.api_key_frame = api_key_frame
        self.rag_frame = None
        self.rag_button = None
        self.back_button = None
        self.chat_display = None
        self.query_input = None
        self.sidebar = None
        self.rag_system = None
        self.setup_buttons()

    def setup_buttons(self):
        button_container = tk.Frame(self.root, bg="#2C3E50")
        button_container.pack(side=tk.BOTTOM, pady=10)

        self.back_button = tk.Button(
            button_container,
            text="Back",
            command=self.switch_to_api_key_frame,
            bg="#34495E",
            fg="#FFFFFF",
            font=("Arial", 10),
            padx=10,
            pady=5
        )
        self.back_button.pack(side=tk.LEFT, padx=5)

        self.rag_button = tk.Button(
            button_container,
            text="RAG",
            command=self.toggle_rag_frame,
            bg="#2980B9",
            fg="#FFFFFF",
            font=("Arial", 10),
            padx=10,
            pady=5
        )
        self.rag_button.pack(side=tk.LEFT, padx=5)

    def toggle_rag_frame(self):
        if self.rag_frame and self.rag_frame.winfo_ismapped():
            self.rag_frame.pack_forget()
            self.main_frame.pack()
        else:
            self.show_rag_frame()

    def show_rag_frame(self):
        global api_key, client

        if not api_key or not client:
            messagebox.showerror("Error", "Set your API key first!")
            return

        self.main_frame.pack_forget()

        if not self.rag_frame:
            self.create_rag_interface()

        self.rag_frame.pack(fill=tk.BOTH, expand=True)

        # Initialize RAG system with API key
        if not self.rag_system:
            try:
                from raggy_hand import RAGSystem
                self.rag_system = RAGSystem(api_key)
                print("RAG System initialized successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to initialize RAG system: {str(e)}")
                self.collapse_rag_and_return()

    def create_rag_interface(self):
        self.rag_frame = tk.Frame(self.root, bg="#1a1a2e")

        # Left sidebar
        self.sidebar = tk.Frame(self.rag_frame, bg="#16213e", width=250)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        # Sidebar title
        tk.Label(
            self.sidebar,
            text="LLM Model",
            bg="#16213e",
            fg="#FFFFFF",
            font=("Arial", 14, "bold")
        ).pack(pady=(20, 10))

        # Model selection dropdown
        self.model_var = tk.StringVar(value="Back")
        model_menu = tk.OptionMenu(
            self.sidebar,
            self.model_var,
            "Back",
            "llama-3.3-70b-versatile",
            "gemma2-9b-it",
            "qwen/qwen3-32b",
            "whisper-large-v3-turbo",
            "meta-llama/llama-prompt-guard-2-22m",
            "llama-3.1-8b-instant"
        )
        model_menu.config(
            bg="#0f3460",
            fg="#FFFFFF",
            highlightbackground="#16213e",
            activebackground="#1a1a2e",
            width=20
        )
        model_menu.pack(padx=10, pady=5, fill=tk.X)

        # Buttons frame
        buttons_frame = tk.Frame(self.sidebar, bg="#16213e")
        buttons_frame.pack(pady=20, fill=tk.X, padx=10)

        # Back button
        back_btn = tk.Button(
            buttons_frame,
            text="Back",
            command=self.collapse_rag_and_return,
            bg="#34495E",
            fg="#FFFFFF",
            font=("Arial", 10),
            width=10,
            pady=5
        )
        back_btn.pack(side=tk.LEFT, padx=5)

        # RAG button
        rag_btn = tk.Button(
            buttons_frame,
            text="RAG",
            command=self.toggle_rag_frame,
            bg="#2980B9",
            fg="#FFFFFF",
            font=("Arial", 10),
            width=10,
            pady=5
        )
        rag_btn.pack(side=tk.LEFT, padx=5)

        # Main content area
        content_frame = tk.Frame(self.rag_frame, bg="#1a1a2e")
        content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Query input section
        query_section = tk.Frame(content_frame, bg="#1a1a2e")
        query_section.pack(fill=tk.X, padx=20, pady=(20, 10))

        tk.Label(
            query_section,
            text="Enter your query:",
            bg="#1a1a2e",
            fg="#FFFFFF",
            font=("Arial", 12)
        ).pack(anchor='w')

        self.query_input = Text(
            query_section,
            height=4,
            width=80,
            bg="#0f3460",
            fg="#FFFFFF",
            insertbackground="#FFFFFF",
            font=("Arial", 10),
            wrap=tk.WORD
        )
        self.query_input.pack(fill=tk.X, pady=5)

        # Right panel with info and buttons
        right_panel = tk.Frame(content_frame, bg="#1a1a2e")
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=20)

        info_frame = tk.Frame(right_panel, bg="#16213e", relief=tk.RAISED, borderwidth=2)
        info_frame.pack(pady=20, padx=10)

        tk.Label(
            info_frame,
            text="Enjoy the retrieval power of\nRAG. I encourage testing\nPDFs first.",
            bg="#16213e",
            fg="#FFFFFF",
            font=("Arial", 11),
            justify="left",
            padx=20,
            pady=20
        ).pack()

        # Document status label
        self.doc_status_label = tk.Label(
            info_frame,
            text="📄 No document loaded",
            bg="#16213e",
            fg="#E74C3C",
            font=("Arial", 9),
            wraplength=180
        )
        self.doc_status_label.pack(pady=(0, 10))

        # Upload Document button
        upload_doc_btn = tk.Button(
            info_frame,
            text="📤 Upload Document",
            command=self.upload_document,
            bg="#27AE60",
            fg="#FFFFFF",
            font=("Arial", 10, "bold"),
            padx=20,
            pady=8,
            cursor="hand2"
        )
        upload_doc_btn.pack(pady=(0, 10), padx=20, fill=tk.X)

        collapse_btn = tk.Button(
            info_frame,
            text="Collapse RAG",
            command=self.collapse_rag_and_return,
            bg="#2980B9",
            fg="#FFFFFF",
            font=("Arial", 10),
            padx=20,
            pady=8
        )
        collapse_btn.pack(pady=(0, 20), padx=20, fill=tk.X)

        # Action buttons
        action_frame = tk.Frame(content_frame, bg="#1a1a2e")
        action_frame.pack(fill=tk.X, padx=20, pady=10)

        submit_btn = tk.Button(
            action_frame,
            text="Submit Query",
            command=self.submit_query,
            bg="#2980B9",
            fg="#FFFFFF",
            font=("Arial", 10),
            padx=20,
            pady=8
        )
        submit_btn.pack(side=tk.LEFT, padx=5)

        use_text_btn = tk.Button(
            action_frame,
            text="Use Extracted Text",
            command=self.use_extracted_text,
            bg="#2980B9",
            fg="#FFFFFF",
            font=("Arial", 10),
            padx=20,
            pady=8
        )
        use_text_btn.pack(side=tk.LEFT, padx=5)

        clear_btn = tk.Button(
            action_frame,
            text="Clear",
            command=self.clear_chat,
            bg="#E74C3C",
            fg="#FFFFFF",
            font=("Arial", 10),
            padx=20,
            pady=8
        )
        clear_btn.pack(side=tk.LEFT, padx=5)

        # Response display label
        tk.Label(
            content_frame,
            text="Response:",
            bg="#1a1a2e",
            fg="#FFFFFF",
            font=("Arial", 12)
        ).pack(anchor='w', padx=20, pady=(20, 5))

        # Response display area
        response_frame = tk.Frame(content_frame, bg="#1a1a2e")
        response_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 10))

        scrollbar = Scrollbar(response_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.chat_display = Text(
            response_frame,
            bg="#0f3460",
            fg="#FFFFFF",
            font=("Arial", 10),
            wrap=tk.WORD,
            state=tk.DISABLED,
            yscrollcommand=scrollbar.set
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.chat_display.yview)

        # PDF upload
        pdf_section = tk.Frame(content_frame, bg="#1a1a2e")
        pdf_section.pack(fill=tk.X, padx=20, pady=(0, 20))

        tk.Label(
            pdf_section,
            text="PDF Text Extractor:",
            bg="#1a1a2e",
            fg="#FFFFFF",
            font=("Arial", 12, "bold")
        ).pack(anchor='w', pady=(0, 10))

        pdf_buttons_frame = tk.Frame(pdf_section, bg="#1a1a2e")
        pdf_buttons_frame.pack(fill=tk.X)

        upload_btn = tk.Button(
            pdf_buttons_frame,
            text="Upload Document",
            command=self.upload_document,
            bg="#2980B9",
            fg="#FFFFFF",
            font=("Arial", 11, "bold"),
            padx=25,
            pady=10,
            relief=tk.RAISED,
            borderwidth=2
        )
        upload_btn.pack(side=tk.LEFT, padx=5)

        view_btn = tk.Button(
            pdf_buttons_frame,
            text="View Extracted Text",
            command=self.view_extracted_text,
            bg="#2980B9",
            fg="#FFFFFF",
            font=("Arial", 11, "bold"),
            padx=25,
            pady=10,
            relief=tk.RAISED,
            borderwidth=2
        )
        view_btn.pack(side=tk.LEFT, padx=5)

        save_btn = tk.Button(
            pdf_buttons_frame,
            text="Save to File",
            command=self.save_to_file,
            bg="#2980B9",
            fg="#FFFFFF",
            font=("Arial", 11, "bold"),
            padx=25,
            pady=10,
            relief=tk.RAISED,
            borderwidth=2
        )
        save_btn.pack(side=tk.LEFT, padx=5)

    def upload_document(self):
        """Upload and process a PDF document for RAG"""
        file_path = filedialog.askopenfilename(
            title="Select PDF Document",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            # Clear previous chat
            self.clear_chat()

            self.update_chat("📄 Processing document...\n", "system")
            self.root.update_idletasks()

            # Initialize RAG system if needed
            if not self.rag_system:
                global api_key
                from raggy_hand import RAGSystem
                self.rag_system = RAGSystem(api_key)

            # Process the document using RAG system
            self.update_chat("📖 Extracting text from PDF...\n", "system")
            self.root.update_idletasks()

            success = self.rag_system.process_document(file_path)

            if success:
                self.update_chat(" Document processed successfully!\n", "system")
                self.update_chat(f" Identified {len(self.rag_system.chapters)} chapters/sections\n", "system")
                self.update_chat(" You can now ask questions about the document.\n\n", "system")

                # Update status label
                self.doc_status_label.config(
                    text=" Document loaded",
                    fg="#27AE60"
                )

                messagebox.showinfo("Success", "PDF processed and ready for queries!")
            else:
                self.update_chat(" Failed to process document.\n", "error")
                messagebox.showerror("Error", "Failed to process the document.")

        except Exception as e:
            error_msg = f"Error: {str(e)}\n"
            self.update_chat(error_msg, "error")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def submit_query(self):
        """Submit a query to the RAG system"""
        if not self.rag_system or not self.rag_system.is_document_loaded():
            messagebox.showwarning("No Document",
                                   "Please upload a PDF document first using the 'Upload Document' button!")
            return

        query = self.query_input.get("1.0", tk.END).strip()

        if not query:
            messagebox.showwarning("Empty Query", "Please enter a query!")
            return

        try:
            self.update_chat(f"\n{'=' * 60}\n", "system")
            self.update_chat(f" Your Query:\n{query}\n\n", "user")
            self.update_chat("🔍 Searching document...\n", "system")
            self.root.update_idletasks()

            # Getting response from RAG system LLAMA 3.3 is used as default
            model = self.model_var.get() if self.model_var.get() != "Back" else "llama-3.3-70b-versatile"

            self.update_chat(f"🤖 Generating response using {model}...\n\n", "system")
            self.root.update_idletasks()

            response = self.rag_system.query(query, model=model)

            self.update_chat(f"💬 Response:\n{response}\n", "assistant")
            self.update_chat(f"\n{'=' * 60}\n\n", "system")

        except Exception as e:
            self.update_chat(f" Error: {str(e)}\n", "error")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def use_extracted_text(self):
        if not self.rag_system or not self.rag_system.is_document_loaded():
            messagebox.showwarning("No Document", "Please upload a PDF document first!")
            return

        extracted_text = self.rag_system.get_full_text()
        current_query = self.query_input.get("1.0", tk.END).strip()

        self.query_input.delete("1.0", tk.END)
        self.query_input.insert("1.0", f"{current_query}\n\n{extracted_text}".strip())

    def view_extracted_text(self):
        if not self.rag_system or not self.rag_system.is_document_loaded():
            messagebox.showwarning("No Document", "Please upload a PDF document first!")
            return

        text_window = Toplevel(self.root)
        text_window.title("Extracted Text")
        text_window.geometry("800x600")
        text_window.configure(bg="#1a1a2e")

        scrollbar = Scrollbar(text_window)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text_display = Text(
            text_window,
            bg="#0f3460",
            fg="#FFFFFF",
            font=("Arial", 10),
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set
        )
        text_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.config(command=text_display.yview)

        text_display.insert(tk.END, self.rag_system.get_full_text())
        text_display.config(state=tk.DISABLED)

    def save_to_file(self):
        if not self.rag_system or not self.rag_system.is_document_loaded():
            messagebox.showwarning("No Document", "Please upload a PDF document first!")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("Markdown files", "*.md")]
        )

        if save_path:
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(self.rag_system.get_full_text())
                messagebox.showinfo("Success", "File saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {str(e)}")

    def clear_chat(self):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.query_input.delete("1.0", tk.END)

    def update_chat(self, message, msg_type="system"):
        self.chat_display.config(state=tk.NORMAL)

        if msg_type == "user":
            self.chat_display.insert(tk.END, message, "user")
        elif msg_type == "assistant":
            self.chat_display.insert(tk.END, message, "assistant")
        elif msg_type == "error":
            self.chat_display.insert(tk.END, message, "error")
        else:
            self.chat_display.insert(tk.END, message)

        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.update_idletasks()

    def collapse_rag_and_return(self):
        if self.rag_frame:
            self.rag_frame.pack_forget()
        self.main_frame.pack()

    def switch_to_api_key_frame(self):
        global api_key, client
        api_key = None
        client = None
        self.main_frame.pack_forget()
        if self.rag_frame:
            self.rag_frame.pack_forget()
        self.api_key_frame.pack()
# API Key Frame
api_key_frame = tk.Frame(main, bg="#2C3E50")
api_key_label = tk.Label(api_key_frame, text="Enter your API Key:", bg="#898a87", font=("Arial", 12))
api_key_label.pack(pady=10)

api_key_entry = tk.Entry(api_key_frame, width=40, show="*")
api_key_entry.pack(pady=10)

api_key_button = tk.Button(api_key_frame, text="Submit API Key", command=set_api_key, bg="#1e162d", fg="#FFFFFF")
api_key_button.pack(pady=10)
api_key_frame.pack()

# Main Frame
main_frame = tk.Frame(main, bg="#2C3E50")
main_frame.pack_forget()

rag_interface = RAGInterface(main, main_frame, api_key_frame)


# Menu Frame with Model Selection
def menu_frame(main):
    global llm_model_var

    container = tk.Frame(main)
    container.pack(side=tk.LEFT, fill=tk.Y, anchor='nw')

    menu_frame = tk.Frame(container, bg="#1B4636", width=220, height=main.winfo_height())
    menu_frame.pack_propagate(False)
    menu_frame.pack(side=tk.LEFT, fill=tk.Y)

    # title
    tk.Label(menu_frame, text="Model Selection", bg="#1B4636", fg="#FFFFFF", font=("Arial", 12, "bold")).pack(pady=10)

    # Dropdown menu options
    llm_model_options = [
        "llama-3.3-70b-versatile",
        "gemma2-9b-it",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "groq/compound",
        "moonshotai/kimi-k2-instruct",
        "qwen/qwen3-32b",
        "openai/gpt-oss-safeguard-20b",
        "allam-2-7b",
        "meta-llama/llama-prompt-guard-2-22m",
    ]

    llm_model_var = tk.StringVar(value=selected_model)
    llm_model_var.trace("w", update_selected_model)

    llm_model = tk.OptionMenu(menu_frame, llm_model_var, *llm_model_options)
    llm_model.config(bg="#34495E", fg="#FFFFFF", highlightbackground="#1B4636", activebackground="#2C3E50")
    llm_model.pack(padx=10, pady=10, fill=tk.X)

    # Chat Concatenation Toggle
    tk.Label(menu_frame, text="Chat Settings", bg="#1B4636", fg="#FFFFFF", font=("Arial", 12, "bold")).pack(
        pady=(20, 10))

    global concat_toggle_button, concat_status_label
    concat_toggle_button = tk.Button(
        menu_frame,
        text="Chat Context: OFF",
        command=toggle_chat_concatenation,
        bg="#E74C3C",
        fg="#FFFFFF",
        font=("Arial", 10)
    )
    concat_toggle_button.pack(padx=10, pady=5, fill=tk.X)

    concat_status_label = tk.Label(
        menu_frame,
        text="Context Mode: Disabled",
        bg="#1B4636",
        fg="#E74C3C",
        font=("Arial", 9)
    )
    concat_status_label.pack(pady=5)

    # Info label
    info_text = tk.Label(
        menu_frame,
        text="Enable context to maintain conversation history across queries.",
        bg="#1B4636",
        fg="#BDC3C7",
        font=("Arial", 8),
        wraplength=200,
        justify="left"
    )
    info_text.pack(padx=10, pady=10)

    toggle_button = tk.Button(container, text="◀", bg="#333", fg="white", command=lambda: toggle_menu())
    toggle_button.pack(side=tk.LEFT, fill=tk.Y)

    def toggle_menu():
        if menu_frame.winfo_ismapped():
            menu_frame.pack_forget()
            toggle_button.config(text="▶")
        else:
            menu_frame.pack(side=tk.LEFT, fill=tk.Y)
            toggle_button.config(text="◀")

    menu_frame.pack_forget()
    return container


# Query Section
query_label = tk.Label(main_frame, text="Enter your query:", bg="#ECECEC", font=("Arial", 11))
query_label.pack(pady=5)

query_entry = Text(main_frame, width=70, height=5, wrap=tk.WORD)
query_entry.pack(pady=5)

toggle_button = tk.Button(main_frame, text="Expand Query Input", command=toggle_query_entry, bg="#1e162d", fg="#FFFFFF")
toggle_button.pack(pady=5)

query_button = tk.Button(main_frame, text="Submit Query", command=get_response, bg="#1e162d", fg="#FFFFFF")
query_button.pack(pady=5)

append_text_button = tk.Button(main_frame, text="Use Extracted Text", command=append_extracted_text, bg="#1e162d",
                               fg="#FFFFFF")
append_text_button.pack(pady=5)

query_display = Text(main_frame, height=10, width=80, bg="#FFFFFF", state=tk.DISABLED, wrap=tk.WORD)
query_display.pack(pady=10)


# Clear and Current buttons
def clear_and_current_buttons(main_frame):
    buttons_frame = tk.Frame(main_frame, bg="#2C3E50")
    buttons_frame.pack(pady=5)

    clear_button = tk.Button(buttons_frame, text="Clear", command=clear_text, bg="#1e162d", fg="#FFFFFF")
    clear_button.pack(side=tk.LEFT, padx=5)

    current_button = tk.Button(buttons_frame, text="Current", command=show_current_response, bg="#1e162d", fg="#FFFFFF")
    current_button.pack(side=tk.LEFT, padx=5)

    history_button = tk.Button(main_frame, text="View History", command=view_history, bg="#3b2b2b", fg="#FFFFFF")
    history_button.pack(pady=5)

    return buttons_frame


# File Handling Section
def doc_handle(main_frame):
    file_label = tk.Label(main_frame, text="PDF Text Extractor:", bg="#ECECEC", font=("Arial", 11))
    file_label.pack(pady=10)

    upload_button = tk.Button(main_frame, text="Upload PDF", command=upload_file, bg="#1e162d", fg="#FFFFFF")
    upload_button.pack(pady=5)

    save_button = tk.Button(main_frame, text="Save to File", command=save_to_file, bg="#1e162d", fg="#FFFFFF")
    save_button.pack(pady=5)

    view_text_button = tk.Button(main_frame, text="View Extracted Text", command=view_text_window, bg="#1e162d",
                                 fg="#FFFFFF")
    view_text_button.pack(pady=5)


clear_and_current_buttons(main_frame)
doc_handle(main_frame)
menu_frame(main)
animation_text(main_frame)

main.mainloop()
