import os
import json
import random
import numpy as np

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq # type: ignore

from rouge_score import rouge_scorer # type: ignore


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RL_DIR = os.path.join(BASE_DIR, "rl_state")
os.makedirs(RL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# LeanContext RL constants (from paper Section 5.6)
# ---------------------------------------------------------------------------
# Actions: threshold ratios for top-k selection (k = action × n_sentences)
RL_ACTIONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
N_CLUSTERS = 8          # number of K-means clusters (state space)
ALPHA = 0.9             # weight balancing accuracy vs cost (paper sets α=0.9)
LEARNING_RATE = 0.1     # Q-table update step size
EPSILON = 0.2           # exploration rate for ε-greedy policy

INPUT_COST_PER_1K = 0.0002
OUTPUT_COST_PER_1K = 0.0002


# ---------------------------------------------------------------------------
# PDF / chunking helpers (unchanged)
# ---------------------------------------------------------------------------

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    return splitter.split_text(text)


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def process_pdf(text, user_id, document_id):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    embeddings = get_embedding_model()
    persist_dir = f"{BASE_DIR}/chroma/user_{user_id}_doc_{document_id}"

    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    vectordb.persist()


# ---------------------------------------------------------------------------
# Embedding utilities
# ---------------------------------------------------------------------------

def _get_embedding(text: str, model: HuggingFaceEmbeddings) -> np.ndarray:
    """Return a 1-D numpy embedding vector for a piece of text."""
    return np.array(model.embed_query(text))

# ---------------------------------------------------------------------------
# Similarity Functions
# ---------------------------------------------------------------------------

def compute_rouge_scores(reference, candidate):
    scorer = rouge_scorer.RougeScorer(
        ['rougeL'],
        use_stemmer=True
    )
    scores = scorer.score(reference, candidate)
    return {
        "rougeL": scores["rougeL"].fmeasure,
    }


def compute_semantic_similarity(text1, text2, embedding_model):
    emb1 = embedding_model.embed_query(text1)
    emb2 = embedding_model.embed_query(text2)

    sim = cosine_similarity(
        np.array(emb1).reshape(1, -1),
        np.array(emb2).reshape(1, -1)
    )[0][0]

    return float(sim)

def calculate_cost(input_tokens, output_tokens):
    input_cost = (input_tokens / 1000) * INPUT_COST_PER_1K
    output_cost = (output_tokens / 1000) * OUTPUT_COST_PER_1K
    return input_cost + output_cost



# ---------------------------------------------------------------------------
# LeanContext: sentence-level top-k selection  (paper Section 5.2)
# ---------------------------------------------------------------------------

def _split_into_sentences(text: str) -> list[str]:
    """Naive sentence splitter suitable for RAG contexts."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]


def _select_top_k_sentences(
    sentences: list[str],
    query_embedding: np.ndarray,
    k: int,
    model: HuggingFaceEmbeddings,
) -> set[int]:
    """
    Return the indices of the top-k sentences most similar to the query.
    Uses cosine similarity between query embedding and each sentence embedding
    (paper eq: Top-k sentences = sort(V, cosine_similarity(v_q, v_si))).
    """
    if not sentences or k <= 0:
        return set()

    sent_embeddings = np.array([_get_embedding(s, model) for s in sentences])
    sims = cosine_similarity(query_embedding.reshape(1, -1), sent_embeddings)[0]
    top_indices = set(np.argsort(sims)[-k:].tolist())
    return top_indices


def _reduce_sentence(sentence: str, ratio: float = 0.80) -> str:
    """
    Lightweight open-source text reduction for less-important sentences.
    Keeps the first (1 - ratio) fraction of words, mimicking the paper's use
    of Selective Context with an 80 % reduction (paper Section 6.3).
    Falls back gracefully for very short sentences.
    """
    words = sentence.split()
    keep = max(1, int(len(words) * (1 - ratio)))
    return " ".join(words[:keep])


def build_reduced_context(
    context: str,
    query: str,
    k: int,
    model: HuggingFaceEmbeddings,
    reduction_ratio: float = 0.80,
) -> tuple[str, float]:
    """
    Implements LeanContext's reduced-context construction (paper Section 5.3):
      1. Split context into sentences.
      2. Identify top-k sentences by cosine similarity with the query.
      3. Keep top-k sentences intact; reduce sentences *between* top-k ones;
         eliminate sentences *after* the last top-k sentence.
      4. Preserve original sentence order.

    Returns:
        reduced_context  – the trimmed context string
        token_ratio τ    – len(reduced_tokens) / len(original_tokens)
    """
    sentences = _split_into_sentences(context)
    n = len(sentences)
    if n == 0:
        return context, 1.0

    # Clamp k to [1, n]
    k = max(1, min(k, n))

    query_embedding = _get_embedding(query, model)
    top_k_indices = _select_top_k_sentences(sentences, query_embedding, k, model)

    if not top_k_indices:
        return context, 1.0

    last_top_k_idx = max(top_k_indices)

    reduced_sentences = []
    for i, sentence in enumerate(sentences):
        if i > last_top_k_idx:
            # Eliminate everything after the last top-k sentence (paper Fig. 4)
            break
        if i in top_k_indices:
            # Keep top-k sentences verbatim
            reduced_sentences.append(sentence)
        else:
            # Reduce less-important sentences between top-k positions
            reduced_sentences.append(_reduce_sentence(sentence, ratio=reduction_ratio))

    reduced_context = " ".join(reduced_sentences)

    original_tokens = len(context.split())
    reduced_tokens = len(reduced_context.split())
    token_ratio = reduced_tokens / original_tokens if original_tokens > 0 else 1.0

    return reduced_context, token_ratio


# ---------------------------------------------------------------------------
# RL State: K-means over (v_c − v_q)  (paper Section 5.6.1)
# ---------------------------------------------------------------------------

def _state_vector(context_embedding: np.ndarray, query_embedding: np.ndarray) -> np.ndarray:
    """State = v_c - v_q  (paper Section 5.6.1, shown best in Table 6)."""
    return context_embedding - query_embedding


def load_kmeans_model(user_id: str):
    """Load a persisted K-means model, or return None if not yet trained."""
    path = os.path.join(RL_DIR, f"user_{user_id}_kmeans.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    km = KMeans(n_clusters=N_CLUSTERS, n_init="auto")
    km.cluster_centers_ = np.array(data["centers"])
    km._n_features_out = km.cluster_centers_.shape[1]
    return km


def save_kmeans_model(user_id: str, km: KMeans):
    path = os.path.join(RL_DIR, f"user_{user_id}_kmeans.json")
    with open(path, "w") as f:
        json.dump({"centers": km.cluster_centers_.tolist()}, f)


def get_state_index(
    context_embedding: np.ndarray,
    query_embedding: np.ndarray,
    km: KMeans | None,
) -> int:
    """Map a (context, query) pair to a discrete state index."""
    sv = _state_vector(context_embedding, query_embedding)
    if km is None:
        # Before K-means is trained, hash into a random bucket
        return random.randint(0, N_CLUSTERS - 1)
    return int(km.predict(sv.reshape(1, -1))[0])


# ---------------------------------------------------------------------------
# RL Q-table persistence  (paper Section 5.5 / Algorithm 1)
# ---------------------------------------------------------------------------

def load_q_table(user_id: str) -> dict:
    """Q-table: {state_index: {action_index: Q-value}}"""
    path = os.path.join(RL_DIR, f"user_{user_id}_qtable.json")
    if os.path.exists(path):
        with open(path) as f:
            return {int(s): {int(a): v for a, v in av.items()}
                    for s, av in json.load(f).items()}
    # Initialise all Q-values to 0
    return {s: {a: 0.0 for a in range(len(RL_ACTIONS))} for s in range(N_CLUSTERS)}


def save_q_table(user_id: str, q_table: dict):
    path = os.path.join(RL_DIR, f"user_{user_id}_qtable.json")
    with open(path, "w") as f:
        json.dump(q_table, f)


# ---------------------------------------------------------------------------
# RL Action selection  (paper Section 5.6.2)
# ---------------------------------------------------------------------------

def select_action(state: int, q_table: dict, epsilon: float = EPSILON) -> int:
    """ε-greedy action selection; returns action index into RL_ACTIONS."""
    if random.random() < epsilon or all(q_table[state][a] == 0 for a in range(len(RL_ACTIONS))):
        return random.randint(0, len(RL_ACTIONS) - 1)
    return max(q_table[state], key=q_table[state].get)


def action_to_k(action_idx: int, n_sentences: int) -> int:
    """Convert action index → concrete k value (k = action × n_sentences)."""
    ratio = RL_ACTIONS[action_idx]
    return max(1, int(ratio * n_sentences))


# ---------------------------------------------------------------------------
# RL Reward  (paper Section 5.6.3)
# ---------------------------------------------------------------------------

def compute_rl_reward(
    token_ratio: float,
    rouge_score: float,
    baseline_rouge: float = 0.5,
    alpha: float = ALPHA,
) -> float:
    """
    R = -(1-α)×τ + α×(2r - r*)
    • Penalises large contexts (high τ).
    • Rewards responses close to (or better than) the baseline ROUGE r*.
    α=0.9 weights accuracy much more heavily than cost (paper Section 5.6.3).
    """
    return -(1 - alpha) * token_ratio + alpha * (2 * rouge_score - baseline_rouge)


def _proxy_rouge(reduced_context: str, original_context: str) -> float:
    """
    Lightweight ROUGE-1 proxy: unigram overlap between reduced and original
    context.  Used at inference time when no ground-truth answer is available.
    """
    orig_words = set(original_context.lower().split())
    red_words = set(reduced_context.lower().split())
    if not orig_words:
        return 0.0
    return len(orig_words & red_words) / len(orig_words)


def update_q_table(
    user_id: str,
    state: int,
    action_idx: int,
    reward: float,
    q_table: dict,
    lr: float = LEARNING_RATE,
):
    """In-place Q-table update and persistence (paper Algorithm 1 line 15)."""
    old_q = q_table[state][action_idx]
    q_table[state][action_idx] = old_q + lr * (reward - old_q)
    save_q_table(user_id, q_table)


# ---------------------------------------------------------------------------
# Main answer function
# ---------------------------------------------------------------------------

def answer_question(question: str, user_id: str, document_id: str, chat_history):
    """
    LeanContext inference (paper Algorithm 2):
      1. Retrieve RAG context from vector DB.
      2. Determine state from (v_c, v_q) via K-means.
      3. Select action (top-k ratio) from Q-table with ε-greedy policy.
      4. Build reduced context: keep top-k sentences, reduce in-between,
         eliminate after last top-k.
      5. Query LLM with reduced context.
      6. Compute reward and update Q-table for online learning.
    """
    embedding_model = get_embedding_model()

    persist_dir = os.path.join(BASE_DIR, "chroma", f"user_{user_id}_doc_{document_id}")
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)

    # ── Step 1: Retrieve RAG context (N=8 chunks, matching paper BBCNews setup) ──
    N_CHUNKS = 4
    docs = vectordb.similarity_search(question, k=N_CHUNKS)

    # Baseline (High Retrieval)
    baseline_docs = vectordb.similarity_search(question, k=6)
    baseline_context = "\n".join(doc.page_content for doc in baseline_docs)

    # Adaptive (Current)
    adaptive_docs = vectordb.similarity_search(question, k=N_CHUNKS)
    adaptive_context = "\n".join(doc.page_content for doc in adaptive_docs)

    if not docs:
        return {
            "answer": "No relevant context found in the document.",
            "followups": [],
        }

    original_context = "\n".join(doc.page_content for doc in docs)
    sentences = _split_into_sentences(original_context)
    n_sentences = len(sentences)

    # ── Step 2: Compute embeddings and state (paper Section 5.6.1) ──
    query_embedding = _get_embedding(question, embedding_model)
    context_embedding = _get_embedding(original_context, embedding_model)

    km = load_kmeans_model(user_id)
    state = get_state_index(context_embedding, query_embedding, km)

    # ── Step 3: Select action via ε-greedy Q-table ──
    q_table = load_q_table(user_id)
    action_idx = select_action(state, q_table, epsilon=EPSILON)
    diagnostic_k = action_to_k(action_idx, n_sentences)


    print(f"[LeanContext] State={state} | Action={RL_ACTIONS[action_idx]:.2f} | k={diagnostic_k}/{n_sentences}")

    # ── Step 4: Build reduced context (paper Section 5.3) ──
    reduced_context, token_ratio = build_reduced_context(
        original_context, question, diagnostic_k, embedding_model
    )

    # ── Step 5: Query LLM ──
    history_text = "".join(
        f"User: {c.question}\nAssistant: {c.answer}\n" for c in chat_history
    )

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    baseline_prompt = f"""
                    You are an expert AI assistant.

                    CONTEXT:
                    {baseline_context}

                    QUESTION:
                    {question}

                    Write a complete detailed answer using only the context.
                    Return only the answer text.
                    """

    baseline_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": baseline_prompt}],
        temperature=0.2,
        max_tokens=1500,
    )

    baseline_answer = baseline_response.choices[0].message.content.strip()

    adaptive_prompt = f"""You are an expert AI assistant. Carefully read the document context and answer the question in full detail.

                CONTEXT:
                {adaptive_context}

                CONVERSATION HISTORY:
                {history_text}

                QUESTION:
                {question}

                INSTRUCTIONS FOR THE ANSWER:
                - Write a thorough, well-structured answer using ONLY the information in the context.
                - Cover every relevant aspect of the question.
                - Use clear paragraphs.
                - Use bullet points or numbered lists where helpful.
                - Include specific names, numbers, methods, tools, or examples from the context.

                After the answer, write:

                ===FOLLOWUPS===

                Then generate exactly 3 NEW follow-up questions that:
                - Are directly related to THIS specific question.
                - Refer to specific entities, skills, methods, or facts mentioned in the answer.
                - Are different in focus (do not ask similar-type questions).
                - Do NOT repeat generic questions like "Can you elaborate more?"
                - Should encourage deeper exploration of the document.

                Each follow-up must be on a new line.
                Return only the answer and follow-ups in this format.
            """

    adaptive_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
                    "role": "system",
                    "content": (
                        "You are a precise, document-grounded AI assistant. "
                        "You answer strictly from the provided context. "
                        "When asked, you generate context-specific follow-up questions."
                    ),
                },
                {"role": "user", "content": adaptive_prompt}],
        temperature=0.5,
        max_tokens=1500,
    )

    raw_output = adaptive_response.choices[0].message.content.strip()
    print(raw_output)

    if "===FOLLOWUPS===" in raw_output:
        answer_part, followup_part = raw_output.split("===FOLLOWUPS===", 1)
        followups = [
            line.strip()
            for line in followup_part.strip().split("\n")
            if line.strip()
        ]
    else:
        answer_part = raw_output
        followups = []

    # ── Step 6: Compute reward and update Q-table (online learning) ──
    proxy_rouge = _proxy_rouge(reduced_context, original_context)
    reward = compute_rl_reward(
        token_ratio=token_ratio,
        rouge_score=proxy_rouge,
        baseline_rouge=0.5,   # expected baseline; tune per domain
        alpha=ALPHA,
    )
    update_q_table(user_id, state, action_idx, reward, q_table)

    context_tokens= len(reduced_context.split())
    original_tokens= len(original_context.split())

    print(f"[LeanContext] τ={token_ratio:.3f} | proxy_rouge={proxy_rouge:.3f} | reward={reward:.3f}")
    print(f"[LeanContext] Context tokens ≈ {context_tokens} "
          f"(original ≈ {original_tokens})")
    
    rouge_scores = compute_rouge_scores(
        baseline_answer,
        answer_part
    )

    semantic_similarity = compute_semantic_similarity(
        baseline_answer,
        answer_part,
        embedding_model
    )

    baseline_usage = baseline_response.usage
    adaptive_usage = adaptive_response.usage

    baseline_input_tokens = baseline_usage.prompt_tokens
    baseline_output_tokens = baseline_usage.completion_tokens

    adaptive_input_tokens = adaptive_usage.prompt_tokens
    adaptive_output_tokens = adaptive_usage.completion_tokens

    baseline_cost = calculate_cost(
        baseline_input_tokens,
        baseline_output_tokens
    )

    adaptive_cost = calculate_cost(
        adaptive_input_tokens,
        adaptive_output_tokens
    )

    cost_saving_percent = abs(
        (baseline_cost - adaptive_cost) / baseline_cost
    ) * 100 if baseline_cost > 0 else 0

    print(cost_saving_percent)


    return {
            "answer": answer_part.strip(),
            "followups": followups,
            "selected_k": diagnostic_k,
            "total_sentences": n_sentences,
            "rl_state": state,
            "rl_action": action_idx,
            "similarity_metrics": {
                    "rougeL": rouge_scores["rougeL"]* 100,
                    "semantic_similarity": semantic_similarity* 100,
                    "context_tokens": context_tokens,
                    "original_tokens": original_tokens,
                    "cost_saving_percent": cost_saving_percent
                }
           }

def update_reward_from_feedback(message, rating):
    """
    Update Q-table using human rating (1–5 stars).
    Integrates human signal into Q-learning.
    """

    if message.rl_state is None or message.rl_action is None:
        return  # Safety

    normalized_rating = rating / 5  # scale to 0–1

    # Convert rating into reward scale (-1 to +1)
    human_reward = (normalized_rating * 2) - 1

    q_table = load_q_table(message.user.id)

    state = message.rl_state
    action = message.rl_action

    old_q = q_table[state][action]

    # Stronger update using human signal
    updated_q = old_q + (0.5 * human_reward)

    q_table[state][action] = updated_q

    save_q_table(message.user.id, q_table)

    print(f"[Human RL] ⭐ Rating={rating} | Reward={human_reward:.3f}")


# ---------------------------------------------------------------------------
# Optional: offline RL training (paper Algorithm 1)
# ---------------------------------------------------------------------------

def train_rl_agent(
    training_samples: list[dict],
    user_id: str,
    n_epochs: int | None = None,
):
    """
    Offline Q-table training (paper Algorithm 1).

    training_samples: list of dicts with keys:
        'context'  – RAG context string
        'question' – user query string
        'answer'   – ground-truth answer string (used for ROUGE reward)

    Steps:
      1. Compute state vectors (v_c - v_q) for all samples.
      2. Fit K-means to obtain cluster centres (states).
      3. For each sample × action, compute reduced context, proxy ROUGE,
         and update the Q-table.
    """
    if not training_samples:
        return

    embedding_model = get_embedding_model()

    # ── Phase 1: Build state vectors and fit K-means (Algorithm 1, lines 1-6) ──
    state_vectors = []
    for sample in training_samples:
        vc = _get_embedding(sample["context"], embedding_model)
        vq = _get_embedding(sample["question"], embedding_model)
        state_vectors.append(_state_vector(vc, vq))

    state_matrix = np.array(state_vectors)
    km = KMeans(n_clusters=N_CLUSTERS, n_init="auto", random_state=42)
    km.fit(state_matrix)
    save_kmeans_model(user_id, km)
    print(f"[LeanContext Train] K-means fitted on {len(state_vectors)} samples.")

    # ── Phase 2: Full exploration to populate Q-table (Algorithm 1, lines 7-16) ──
    q_table = load_q_table(user_id)
    epochs = n_epochs or (len(training_samples) * len(RL_ACTIONS))

    for epoch in range(epochs):
        sample = training_samples[epoch % len(training_samples)]
        context = sample["context"]
        question = sample["question"]
        ground_truth = sample.get("answer", "")

        vc = _get_embedding(context, embedding_model)
        vq = _get_embedding(question, embedding_model)
        state = int(km.predict(_state_vector(vc, vq).reshape(1, -1))[0])

        # Full exploration: cycle through all actions
        action_idx = epoch % len(RL_ACTIONS)
        sentences = _split_into_sentences(context)
        k = action_to_k(action_idx, len(sentences))

        reduced_context, token_ratio = build_reduced_context(
            context, question, k, embedding_model
        )

        # ROUGE proxy against ground truth when available, else vs original context
        ref = ground_truth if ground_truth else context
        proxy_rouge = _proxy_rouge(reduced_context, ref)

        reward = compute_rl_reward(token_ratio, proxy_rouge, alpha=ALPHA)
        update_q_table(user_id, state, action_idx, reward, q_table)

    print(f"[LeanContext Train] Q-table updated over {epochs} epochs.")