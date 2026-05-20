# CloudOps AI Assistant ☁️🤖

CloudOps AI Assistant is a modern Retrieval-Augmented Generation (RAG) web application focused on DevOps and Cloud Infrastructure knowledge.

The application allows users to upload DevOps-related documents and ask AI-powered questions about Docker, Kubernetes, Linux, CI/CD pipelines, cloud infrastructure, monitoring, and networking.

The system uses semantic search with FAISS vector retrieval and Hugging Face embeddings to retrieve relevant document chunks before generating grounded responses with Gemini.

---

# 🚀 Features

* 🔍 Semantic document retrieval using FAISS
* 🤖 AI-powered answers with Gemini
* 📄 Upload PDF and TXT documents
* 🧠 Conversation memory with SQLite
* ⚡ Real-time DevOps Q&A experience
* 📚 Retrieved source display with similarity scores
* 🎨 Modern DevOps-inspired UI
* 🌙 Dark futuristic dashboard design
* ☁️ Cloud & infrastructure knowledge base

---

# 🏗️ RAG Architecture

```text
User Question
      ↓
Hugging Face Embeddings
      ↓
FAISS Vector Search
      ↓
Relevant Chunks Retrieved
      ↓
Prompt Augmentation
      ↓
Gemini LLM
      ↓
Grounded AI Response
```

---

# 🧰 Technologies Used

## Backend

* Python
* Flask
* SQLite

## AI / RAG Stack

* FAISS
* Hugging Face Inference API
* Gemini 2.5 Flash
* Sentence Embeddings

## Frontend

* HTML5
* CSS3
* Vanilla JavaScript

## Document Processing

* PyPDF2
* NLTK

---

# 📂 Project Structure

```text
cloudops-ai-assistant/
│
├── app.py
├── rag_example.py
├── requirements.txt
├── README.md
├── .env
├── .gitignore
│
├── data/
│   ├── docker_basics.txt
│   ├── kubernetes.txt
│   ├── cicd_pipeline.txt
│   ├── linux_and_networking.txt
│   └── cloud_and_monitoring.txt
│
├── templates/
│   ├── base.html
│   └── index.html
│
├── static/
│   ├── css/
│   │   └── main.css
│   │
│   └── js/
│       └── app.js
│
└── uploads/
```

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/yourusername/cloudops-ai-assistant.git

cd cloudops-ai-assistant
```

---

## 2. Create virtual environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key

HF_TOKEN=your_huggingface_token
```

---

# ▶️ Running the Application

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

---

# 📄 Supported File Types

* PDF
* TXT

---

# 🧠 Example Questions

```text
What is the difference between a Kubernetes pod and deployment?

Explain Docker networking.

What are liveness probes?

How does CI/CD work?

What is horizontal scaling?

What does chmod do in Linux?
```

---

# 🔍 Retrieval Pipeline

1. Documents are uploaded and processed
2. Text is split into chunks
3. Chunks are embedded using Hugging Face embeddings
4. Embeddings are stored inside a FAISS vector database
5. User questions are embedded
6. Relevant chunks are retrieved semantically
7. Gemini generates grounded responses using retrieved context

---

# 🛡️ Hallucination Prevention

The system reduces hallucinations by:

* Using retrieved document context
* Applying semantic similarity filtering
* Rejecting low-relevance matches
* Instructing the LLM to answer only from retrieved knowledge

If the documents do not contain enough information, the assistant responds with:

```text
"I do not have enough information in the documents..."
```

---

# 🧪 Validation & Testing

The application was tested using multiple DevOps and Cloud Infrastructure questions to validate:

* Accurate semantic retrieval
* Context-grounded answering
* Conversation memory
* Irrelevant query handling
* Source visibility and similarity scoring

---

# 📈 Future Improvements

* Streaming AI responses
* Docker deployment
* Kubernetes deployment
* User authentication
* Persistent vector databases
* Multi-user support
* Hybrid search
* LangChain integration

---

# 👨‍💻 Author

Lee Ben Shimon

B.Sc. Computer Science Student
Fullstack & DevOps Enthusiast

---

# 📜 License

This project was created for educational purposes.
