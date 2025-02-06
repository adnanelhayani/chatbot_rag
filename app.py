from flask import Flask, request, jsonify, render_template
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
import os
from flask_cors import CORS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Configuration de MongoDB et Flask
app = Flask(__name__)
CORS(app, supports_credentials=True)

GOOGLE_API_KEY = 'AIzaSyDlny_qz8tHGX2NfxWfou1SYaaxV_wPSUA'
genai.configure(api_key=GOOGLE_API_KEY)

# Configuration de Chroma pour la persistance
CHROMA_DB_DIRECTORY = "./chroma_db"

# Utilisation du modèle de génération Google
model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=0.2)

# Charger et diviser plusieurs PDF à partir d'un répertoire
pdf_directory = "./cybersecurity_files" 
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

pages = []
for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    pdf_loader = PyPDFLoader(pdf_path)
    pages.extend(pdf_loader.load_and_split())

# Préparer le texte pour l'intégration avec les embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
context = "\n\n".join(str(p.page_content) for p in pages)
texts = text_splitter.split_text(context)

# Embedding avec Google Generative AI
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

if not os.path.exists(CHROMA_DB_DIRECTORY):
    os.makedirs(CHROMA_DB_DIRECTORY)
    vector_index = Chroma.from_texts(texts, embeddings, persist_directory=CHROMA_DB_DIRECTORY)
    vector_index.persist()
else:
    vector_index = Chroma(persist_directory=CHROMA_DB_DIRECTORY, embedding_function=embeddings)

template = """
Respond only in English. If the context does not answer the question, 
please speculate and mention that you are unsure. Ask for more details if needed.

{context}

---

Answer the following question using the context above: {question}
Réponse utile :"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    model,
    retriever=vector_index.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get('question')

    if user_input:
        user_input_lower = user_input.lower()
        if user_input_lower in ["hi", "hello"]:
            response_text = "Hello! I'm CyberGuard. How can I assist you with cybersecurity today?"
            return jsonify({'response': response_text, 'sources': []})
        elif "what is cyberguard" in user_input_lower:
            response_text = (
                "CyberGuard is an innovative cybersecurity chatbot developed by five talented second-year Master's students in Cybersecurity and Big Data at FSTT: "
                "AKZOUN Hafsa, BOULBEN Firdaous, EL HAYANI Adnan, EL YAHYAOUY Imane, and TOUYEB Zakaria. "
                "It integrates advanced Language Understanding Models (LUM), Diffusion Models for threat prediction, and Retrieval-Augmented Generation (RAG) "
                "to provide real-time, intelligent security solutions within a scalable microservices architecture."
            )
            return jsonify({'response': response_text, 'sources': []})
        else:
            result = qa_chain.invoke({"query": user_input})
            source_documents = [str(doc) for doc in result['source_documents']]
            response_text = result['result'][:900]
            return jsonify({'response': response_text, 'sources': source_documents})
    return jsonify({'error': 'No question provided'}), 400

@app.route('/')
def index():
    return render_template('indexx.html')

if __name__ == '__main__':
    app.run(debug=True)
