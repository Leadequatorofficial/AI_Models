from fastapi import FastAPI
from pydantic import BaseModel
from keybert import KeyBERT
from fastapi.middleware.cors import CORSMiddleware
import re


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def home():
    return {"message": "AI Keyword Extractor API is running"}

kw_model = None

def get_model():
    global kw_model
    if kw_model is None:
        print("Loading KeyBERT model...")
        kw_model = KeyBERT()
    return kw_model


# -------- INPUT SCHEMA --------
from typing import Optional

class FormData(BaseModel):
    name: Optional[str] = ""
    job_title: Optional[str] = ""
    industry: Optional[str] = ""
    company_type: Optional[str] = ""
    interests: Optional[str] = ""
    location: Optional[str] = ""
    problem: Optional[str] = ""

# -------- CLEAN TEXT --------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def extract_keyword_strings(keywords):
    return [kw for kw, score in keywords]

def clean_keywords(keywords):
    cleaned = set()
    for kw, _ in keywords:
        words = kw.split()
        if len(words) >= 2:
            if words.count(words[-1]) == 1:  # remove "export export"
                cleaned.add(kw)
    return list(cleaned)

# -------- QUORA QUESTION GENERATOR --------
def generate_quora_questions(keyword_strings):
    questions = []
    for kw in keyword_strings:
        questions.extend([
            f"How can I find reliable {kw}?",
            f"What are the best strategies for {kw}?",
            f"How do beginners start with {kw}?",
            f"What problems do people face in {kw}?"
        ])
    return list(set(questions))



# -------- REDDIT SEARCH & SUBREDDITS --------
def generate_reddit_content(keyword_strings):
    queries = []
    subreddits = set()

    for kw in keyword_strings:
        queries.extend([
            f"{kw} advice",
            f"{kw} problems",
            f"anyone experienced with {kw}"
        ])

        if "export" in kw:
            subreddits.update(["r/export", "r/logistics"])
        if "agriculture" in kw or "produce" in kw or "fruits" in kw:
            subreddits.update(["r/agriculture", "r/farming"])
        if "business" in kw or "market" in kw:
            subreddits.add("r/Entrepreneur")

    return list(set(queries)), list(subreddits)


# -------- API ENDPOINT --------
@app.post("/extract-keywords")
def extract_keywords(data: FormData):

    combined_text = " ".join([
    data.industry,
    data.company_type,
    data.interests,
    data.problem,
    data.location
])


    cleaned_text = clean_text(combined_text)

    model = get_model()

    keywords = model.extract_keywords(
        cleaned_text,
        keyphrase_ngram_range=(1, 3),
        stop_words='english',
        top_n=7
    )

    keyword_strings = extract_keyword_strings(keywords)
    quora_questions = generate_quora_questions(keyword_strings)
    reddit_queries, subreddits = generate_reddit_content(keyword_strings)


    return {
        "core_keywords": [kw for kw, _ in keywords],
        "quora_questions": quora_questions[:8],
        "reddit_search_queries": reddit_queries[:8],
        "recommended_subreddits": subreddits
    }
