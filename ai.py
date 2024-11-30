import tkinter as tk
from tkinter import scrolledtext, Entry, Button, END
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

DB_DIR = "./vector_db/"
MODEL_NAME = "gemini-pro"
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
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

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
        return self.qa_chain.invoke({"query": question})

class ChatbotGUI:
    def __init__(self, qa_system: QASystem):
        self.qa_system = qa_system
        self.window = tk.Tk()
        self.window.title("RAG Chatbot")
        self.window.geometry("600x400")

        self.chat_history = scrolledtext.ScrolledText(self.window, wrap=tk.WORD, width=70, height=20)
        self.chat_history.pack(padx=10, pady=10)
        self.chat_history.config(state=tk.DISABLED)

        self.user_input = Entry(self.window, width=50)
        self.user_input.pack(side=tk.LEFT, padx=10)

        self.send_button = Button(self.window, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.LEFT)

        self.user_input.bind("<Return>", lambda event: self.send_message())

    def send_message(self):
        user_message = self.user_input.get()
        if user_message.strip() != "":
            self.display_message("You: " + user_message)
            self.user_input.delete(0, END)

            try:
                result = self.qa_system.answer_question(user_message)
                answer = result["result"]
                sources = result["source_documents"][:2]

                self.display_message("Chatbot: " + answer)
                self.display_message("\nSources:")
                for doc in sources:
                    self.display_message(doc.page_content[:200] + "...")
            except Exception as e:
                self.display_message(f"An error occurred: {str(e)}")

    def display_message(self, message):
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.insert(tk.END, message + "\n\n")
        self.chat_history.see(tk.END)
        self.chat_history.config(state=tk.DISABLED)

    def run(self):
        self.window.mainloop()

def main():
    qa_system = QASystem(DB_DIR)
    chatbot_gui = ChatbotGUI(qa_system)
    chatbot_gui.run()

if __name__ == "__main__":
    main()