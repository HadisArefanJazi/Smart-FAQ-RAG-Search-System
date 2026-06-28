from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


DATA_PATH = Path("data/raw/faqs.csv")


sample_faqs = [
    {
        "id": 1,
        "question": "How can I reset my password?",
        "answer": "You can reset your password by clicking Forgot Password on the login page.",
        "category": "account"
    },
    {
        "id": 2,
        "question": "How can I contact support?",
        "answer": "You can contact support through the help center or by emailing support@example.com.",
        "category": "support"
    },
    {
        "id": 3,
        "question": "Can I cancel my subscription?",
        "answer": "You can cancel your subscription from your account billing settings.",
        "category": "billing"
    }
]


def clean_text(text):
    return " ".join(str(text).lower().strip().split())


def load_faqs():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.DataFrame(sample_faqs)

    required_columns = {"id", "question", "answer", "category"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    return df


def make_faq_chunks(df):
    chunks = []

    for _, row in df.iterrows():
        chunks.append({
            "id": int(row["id"]),
            "source": row["category"],
            "question": row["question"],
            "answer": row["answer"],
            "text": f"{row['question']} {row['answer']}"
        })

    return chunks


faqs = load_faqs()
chunks = make_faq_chunks(faqs)


def search_tfidf(question, top_k=3):
    texts = [clean_text(chunk["text"]) for chunk in chunks]

    vectorizer = TfidfVectorizer(stop_words="english")
    chunk_vectors = vectorizer.fit_transform(texts)
    question_vector = vectorizer.transform([clean_text(question)])

    scores = cosine_similarity(question_vector, chunk_vectors)[0]
    ranked_indexes = scores.argsort()[::-1]

    results = []

    for index in ranked_indexes[:top_k]:
        results.append({
            "id": chunks[index]["id"],
            "source": chunks[index]["source"],
            "question": chunks[index]["question"],
            "answer": chunks[index]["answer"],
            "score": float(scores[index])
        })

    return results


def search_semantic(question, top_k=3):
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is not installed. Run: pip install sentence-transformers")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [chunk["text"] for chunk in chunks]

    chunk_vectors = model.encode(texts)
    question_vector = model.encode([question])

    scores = cosine_similarity(question_vector, chunk_vectors)[0]
    ranked_indexes = scores.argsort()[::-1]

    results = []

    for index in ranked_indexes[:top_k]:
        results.append({
            "id": chunks[index]["id"],
            "source": chunks[index]["source"],
            "question": chunks[index]["question"],
            "answer": chunks[index]["answer"],
            "score": float(scores[index])
        })

    return results


def rerank(question, results):
    words = clean_text(question).split()

    for result in results:
        text = clean_text(result["question"] + " " + result["answer"])
        bonus = 0

        for word in words:
            if word in text:
                bonus += 0.05

        result["rerank_score"] = result["score"] + bonus

    return sorted(results, key=lambda item: item["rerank_score"], reverse=True)


def make_prompt(question, results):
    context = ""

    for result in results:
        context += f"Source: {result['source']}\n"
        context += f"FAQ Question: {result['question']}\n"
        context += f"FAQ Answer: {result['answer']}\n\n"

    prompt = f"""
Use only the FAQ context below to answer the question.

If the answer is not in the FAQ context, say:
I do not have enough information in the FAQ data.

FAQ Context:
{context}

User Question:
{question}

Answer:
"""

    return prompt.strip()


def answer_faq(question, method="tfidf", top_k=3, threshold=0.20):
    if method == "tfidf":
        results = search_tfidf(question, top_k)
    elif method == "semantic":
        results = search_semantic(question, top_k)
    else:
        raise ValueError("method must be 'tfidf' or 'semantic'")

    results = rerank(question, results)
    best = results[0]
    prompt = make_prompt(question, results)

    if best["score"] < threshold:
        answer = "I do not have enough information in the FAQ data."
        sources = []
    else:
        answer = best["answer"]
        sources = [best["source"]]

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "best_score": round(best["score"], 3),
        "retrieved_faqs": results,
        "prompt": prompt
    }


def print_response(response):
    print("=" * 70)
    print("Question:")
    print(response["question"])

    print("\nAnswer:")
    print(response["answer"])

    print("\nSources:")
    print(response["sources"])

    print("\nBest score:")
    print(response["best_score"])

    print("\nRetrieved FAQs:")

    for faq in response["retrieved_faqs"]:
        print("-" * 50)
        print("Source:", faq["source"])
        print("Question:", faq["question"])
        print("Answer:", faq["answer"])
        print("Score:", round(faq["score"], 3))


test_questions = [
    {
        "question": "how do I reset my password?",
        "expected_source": "account"
    },
    {
        "question": "how do I contact support?",
        "expected_source": "support"
    },
    {
        "question": "how do I cancel billing?",
        "expected_source": "billing"
    }
]


def evaluate(method="tfidf"):
    correct = 0

    for test in test_questions:
        response = answer_faq(test["question"], method=method)

        predicted_source = None

        if response["sources"]:
            predicted_source = response["sources"][0]

        if predicted_source == test["expected_source"]:
            correct += 1

        print("-" * 70)
        print("Question:", test["question"])
        print("Expected:", test["expected_source"])
        print("Predicted:", predicted_source)

    accuracy = correct / len(test_questions)

    print("\nAccuracy:", round(accuracy, 2))


def add_faq(question, answer, category):
    global faqs
    global chunks

    new_id = int(faqs["id"].max()) + 1

    new_row = pd.DataFrame([{
        "id": new_id,
        "question": question,
        "answer": answer,
        "category": category
    }])

    faqs = pd.concat([faqs, new_row], ignore_index=True)
    chunks = make_faq_chunks(faqs)


if __name__ == "__main__":
    questions = [
        "how do I reset my password?",
        "can I cancel my subscription?",
        "how can I contact support?",
        "what is the refund policy?"
    ]

    for question in questions:
        response = answer_faq(question, method="tfidf")
        print_response(response)

    print("\nGrounded Prompt Example:")
    response = answer_faq("how do I reset my password?", method="tfidf")
    print(response["prompt"])

    print("\nEvaluation:")
    evaluate(method="tfidf")

    print("\nSemantic search is optional.")
    print("Install it with:")
    print("pip install sentence-transformers")

    print("\nThen test:")
    print("answer_faq('how do I cancel my plan?', method='semantic')")
