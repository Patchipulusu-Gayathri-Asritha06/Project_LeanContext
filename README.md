# 🚀 LeanContext

Adaptive Reinforcement Learning based Context Compression for RAG Systems.

LeanContext is a Django-based Retrieval-Augmented Generation (RAG) system that dynamically selects optimal context size using Reinforcement Learning.

It balances:

- ✅ Answer Accuracy (ROUGE + Semantic Similarity)
- ✅ Token Cost Reduction
- ✅ Human Feedback (1–5 star rating as RL reward)
- ✅ Adaptive Top-K Selection using Q-Learning

---

# 📌 Features

- 🔍 RAG using ChromaDB + HuggingFace Embeddings
- 🧠 RL-based dynamic context compression
- ⭐ Human feedback integrated into Q-table
- 📊 Similarity metrics (ROUGE-L + Semantic Similarity)
- 💰 Cost savings calculation
- 📄 PDF upload and processing
- 👤 User authentication system
- 🗂 Session-based chat history

---

# 🛠 Tech Stack

- Django 5
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Groq LLM (LLaMA 3)
- Scikit-learn (KMeans)
- ROUGE Score
- Reinforcement Learning (Q-learning)

---

# ⚙️ Installation Guide

Follow these steps after cloning the repository.

---

## 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/LeanContext.git
cd LeanContext
```

---

## 2️⃣ Create Virtual Environment

### Windows
```bash
python -m venv myEnv
myEnv\Scripts\activate
```

### Mac / Linux
```bash
python3 -m venv myEnv
source myEnv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4️⃣ Create Environment Variables (.env)

Create a file in the root directory:

```
.env
```

Add the following variables:

```
SECRET_KEY=your_django_secret_key
DEBUG=True

GROQ_API_KEY=your_groq_api_key

EMAIL_HOST_USER=your_email@gmail.com
EMAIL_HOST_PASSWORD=your_app_password
```

### 🔑 How To Generate Django Secret Key

You can generate one using:

```bash
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

### 🔑 Groq API Key

Get it from:
https://console.groq.com/

### 🔑 Gmail App Password

Go to: https://myaccount.google.com/apppasswords
1. Enable 2-Step Verification
2. Generate App Password from Google Account
3. Use that password in `.env`

---

## 5️⃣ Run Database Migrations

```bash
python manage.py migrate
```

---

## 6️⃣ Create Superuser (Optional)

```bash
python manage.py createsuperuser
```

---

## 7️⃣ Run Development Server

```bash
python manage.py runserver
```

Open in browser:

```
http://127.0.0.1:8000/
```

---

# 🧠 How RL Works

1. Retrieve context from ChromaDB
2. Compute state = (context_embedding − query_embedding)
3. Map state using KMeans clustering
4. Select action via ε-greedy Q-table
5. Reduce context using top-k sentences
6. Generate answer
7. Compute reward:
   - Token ratio penalty
   - Proxy ROUGE
   - Human rating
8. Update Q-table

---

# 📂 Auto-Generated Folders

The following folders are created automatically:

```
media/              → uploaded PDFs
ragapp/chroma/      → vector database
ragapp/rl_state/    → RL Q-tables
```

These are excluded from Git.

---

# 🔐 Security Notes

- `.env` is excluded from Git
- Do NOT commit API keys
- Regenerate email passwords if exposed
- Set DEBUG=False in production

---

# 📊 Project Structure

```
LeanContext/
│
├── LeanContext/        # Django project settings
├── ragapp/             # RAG + RL logic
├── users/              # Authentication app
├── manage.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🚀 Running After Fresh Clone (Quick Commands)

```bash
git clone https://github.com/YOUR_USERNAME/LeanContext.git
cd LeanContext
python -m venv myEnv
myEnv\Scripts\activate   # Windows
pip install -r requirements.txt
# Create .env file
python manage.py migrate
python manage.py runserver
```

---

# 📈 Future Improvements

- Deploy to AWS / Render
- Replace SQLite with PostgreSQL
- Add reward visualization dashboard
- Add model comparison mode
- Convert to API (DRF)

---

# 👩‍💻 Author

Patchipulusu Gayathri Asritha 
LeanContext – Reinforcement Learning Driven Context Compression for RAG

---
