import tkinter as tk
from tkinter import scrolledtext, Entry, Button, END, Frame, Label
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import warnings
import time

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

# Constants
DB_DIR = "./vector_db/"
MODEL_NAME = "gemini-1.5-pro"
TEMPERATURE = 0.3

class QASystem:
    def __init__(self, db_dir: str):
        self.db_dir = db_dir
        self.vector_store = self.load_vector_store()
        self.qa_chain = self.create_qa_chain()

    def load_vector_store(self) -> Chroma:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        return Chroma(persist_directory=self.db_dir, embedding_function=embeddings)

    def create_qa_chain(self) -> RetrievalQA:
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=TEMPERATURE)
        
        prompt_template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you dont know the answer.
        {context}
        Question: {question}
        Helpful Answer:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        chain_type_kwargs = {"prompt": PROMPT}
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True
        )

    def answer_question(self, question: str) -> Dict[str, Any]:
        # Time the vector search
        search_start = time.time()
        docs = self.vector_store.similarity_search(question)
        search_time = (time.time() - search_start) * 1000  # Convert to milliseconds
        
        # Time the response generation
        generation_start = time.time()
        result = self.qa_chain.invoke({"query": question})
        generation_time = (time.time() - generation_start) * 1000  # Convert to milliseconds
        
        result['search_time'] = search_time
        result['generation_time'] = generation_time
        return result

class ChatbotGUI:
    def __init__(self, qa_system: QASystem):
        self.qa_system = qa_system
        self.window = tk.Tk()
        self.window.title("RAG Chatbot")
        self.window.geometry("800x600")
        self.window.configure(bg="#2b2b2b")

        self.create_widgets()

    def create_widgets(self):
        # Main frame
        main_frame = Frame(self.window, bg="#2b2b2b")
        main_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Chat frame (left side)
        chat_frame = Frame(main_frame, bg="#2b2b2b")
        chat_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        # Chat history
        self.chat_history = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, width=50, height=25, bg="#3b3b3b", fg="white")
        self.chat_history.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        self.chat_history.config(state=tk.DISABLED)

        # Input frame
        input_frame = Frame(chat_frame, bg="#2b2b2b")
        input_frame.pack(fill=tk.X, padx=5, pady=5)

        self.user_input = Entry(input_frame, bg="#4a4a4a", fg="white", insertbackground="white")
        self.user_input.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.send_button = Button(input_frame, text="Send", command=self.send_message, bg="#5a5a5a", fg="white", activebackground="#6a6a6a", activeforeground="white")
        self.send_button.pack(side=tk.RIGHT, padx=(5, 0))

        self.user_input.bind("<Return>", lambda event: self.send_message())

        # Right side frame
        right_frame = Frame(main_frame, bg="#2b2b2b")
        right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Sources section
        sources_label = Label(right_frame, text="Sources", bg="#2b2b2b", fg="white", font=("Arial", 12, "bold"))
        sources_label.pack(pady=(0, 5))

        self.sources_text = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, width=30, height=20, bg="#3b3b3b", fg="white")
        self.sources_text.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        self.sources_text.config(state=tk.DISABLED)

        # Benchmarks section
        benchmarks_frame = Frame(right_frame, bg="#2b2b2b")
        benchmarks_frame.pack(fill=tk.X, padx=5, pady=5)

        # Vector Search Time
        search_frame = Frame(benchmarks_frame, bg="#2b2b2b")
        search_frame.pack(fill=tk.X, pady=2)
        Label(search_frame, text="Vector Search:", bg="#2b2b2b", fg="white").pack(side=tk.LEFT)
        self.search_time_label = Label(search_frame, text="0 ms", bg="#2b2b2b", fg="#4CAF50")
        self.search_time_label.pack(side=tk.RIGHT)

        # Response Generation Time
        generation_frame = Frame(benchmarks_frame, bg="#2b2b2b")
        generation_frame.pack(fill=tk.X, pady=2)
        Label(generation_frame, text="Response Generation:", bg="#2b2b2b", fg="white").pack(side=tk.LEFT)
        self.generation_time_label = Label(generation_frame, text="0 ms", bg="#2b2b2b", fg="#4CAF50")
        self.generation_time_label.pack(side=tk.RIGHT)

        # Configure text tags for colors
        self.chat_history.tag_configure("user", foreground="#4CAF50")
        self.chat_history.tag_configure("bot", foreground="#2196F3")
        self.chat_history.tag_configure("system", foreground="#FFC107")

    def send_message(self):
        user_message = self.user_input.get()
        if user_message.strip() != "":
            self.display_message("You: " + user_message, "user")
            self.user_input.delete(0, END)

            try:
                result = self.qa_system.answer_question(user_message)
                answer = result["result"]
                sources = result["source_documents"][:2]
                search_time = result["search_time"]
                generation_time = result["generation_time"]

                self.display_message("Chatbot: " + answer, "bot")
                self.display_sources(sources)
                self.update_benchmarks(search_time, generation_time)
            except Exception as e:
                self.display_message(f"An error occurred: {str(e)}", "system")

    def display_message(self, message, tag):
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.insert(tk.END, message + "\n\n", tag)
        self.chat_history.see(tk.END)
        self.chat_history.config(state=tk.DISABLED)

    def display_sources(self, sources):
        self.sources_text.config(state=tk.NORMAL)
        self.sources_text.delete('1.0', END)
        for i, doc in enumerate(sources, 1):
            self.sources_text.insert(END, f"Source {i}:\n{doc.page_content[:200]}...\n\n")
        self.sources_text.config(state=tk.DISABLED)

    def update_benchmarks(self, search_time: float, generation_time: float):
        self.search_time_label.config(text=f"{search_time:.2f} ms")
        self.generation_time_label.config(text=f"{generation_time:.2f} ms")

    def run(self):
        self.window.mainloop()

def main():
    qa_system = QASystem(DB_DIR)
    chatbot_gui = ChatbotGUI(qa_system)
    chatbot_gui.run()

if __name__ == "__main__":
    main()