from flask import Flask, request, jsonify, render_template
from langchain_community.vectorstores import Chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
import os
import textwrap
from flask import Flask, request, jsonify, render_template
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from werkzeug.security import generate_password_hash
from flask_cors import CORS  # Importez CORS
from flask import Flask, jsonify, request
from werkzeug.security import generate_password_hash, check_password_hash

from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from werkzeug.security import generate_password_hash, check_password_hash
import pymongo
import jwt
from datetime import datetime, timedelta
from bson.objectid import ObjectId
import textwrap
from rapidfuzz import process  # Pour la recherche floue

# client = pymongo.MongoClient("mongodb://localhost:27017/")
# db = client["database1"]  # Remplacez par le nom de votre base de données
# user_collection = db["users"]

# SECRET_KEY = 'test'

# Configuration de MongoDB et Flask
app = Flask(__name__)
CORS(app, supports_credentials=True)

GOOGLE_API_KEY = 'AIzaSyDlny_qz8tHGX2NfxWfou1SYaaxV_wPSUA'
genai.configure(api_key=GOOGLE_API_KEY)

# Configuration de Chroma pour la persistance
CHROMA_DB_DIRECTORY = "./chroma_db"  # Répertoire où les données seront stockées

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

# Vérifiez si une base de données Chroma existe déjà, sinon créez-en une
if not os.path.exists(CHROMA_DB_DIRECTORY):
    os.makedirs(CHROMA_DB_DIRECTORY)
    # Créer une nouvelle base de données Chroma et y ajouter les textes
    vector_index = Chroma.from_texts(texts, embeddings, persist_directory=CHROMA_DB_DIRECTORY)
    vector_index.persist()  # Enregistrer la base de données sur le disque
else:
    # Charger la base de données Chroma existante
    vector_index = Chroma(persist_directory=CHROMA_DB_DIRECTORY, embedding_function=embeddings)

# Modèle de question et de réponse pour la cybersécurité
# template = """Utilisez les éléments de contexte suivants pour répondre à la question à la fin. Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas, n'essayez pas d'inventer une réponse. 
# Renvoyez une réponse sous forme de phrases complètes et pertinentes en vous concentrant sur les bonnes pratiques en cybersécurité. Terminez toujours la réponse par "Merci pour votre question !".
# {context}
# Question : {question}
# Réponse utile :"""
template = """
Respond only in English. If the context does not answer the question, 
please speculate and mention that you are unsure. Ask for more details if needed.

{context}

---

Answer the following question using the context above: {question}
Réponse utile :"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

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
        # Interroger la chaîne de récupération avec la question de l'utilisateur
        result = qa_chain.invoke({"query": user_input})

        # Convertir les documents source en chaînes de caractères
        source_documents = [str(doc) for doc in result['source_documents']]

        # Limiter la réponse pour éviter des longueurs excessives
        response_text = result['result'][:900]  # Limiter la longueur de la réponse pour la cybersécurité

        # Renvoi de la réponse
        return jsonify({
            'response': response_text,
            'sources': source_documents
        })
    return jsonify({'error': 'No question provided'}), 400







@app.route('/')
def index():
    return render_template('indexx.html')


if __name__ == '__main__':
    app.run(debug=True)

