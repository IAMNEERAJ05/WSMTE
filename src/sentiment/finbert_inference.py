"""
src/sentiment/finbert_inference.py
FinBERT polarity and mDeBERTa subjectivity inference.
Runs on GPU (Kaggle T4 x2). Not needed locally after outputs are saved.
"""

import re
import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_finbert(config, device=None):
    """Load ProsusAI/finbert tokenizer and model. Returns (tokenizer, model, device)."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained(config['finbert_model'])
    model = BertForSequenceClassification.from_pretrained(config['finbert_model'])
    model.eval()
    model = model.to(device)
    print(f"FinBERT loaded on {device}")
    return tokenizer, model, device


def load_mdeberta(config, device=None):
    """Load cross-encoder/nli-deberta-v3-small tokenizer and model."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(config['mdeberta_model'])
    model = AutoModelForSequenceClassification.from_pretrained(config['mdeberta_model'])
    model.eval()
    model = model.to(device)
    print(f"mDeBERTa loaded on {device}")
    return tokenizer, model, device


# ─────────────────────────────────────────────────────────────────────────────
# Text preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_kotekar_text(row, n_sentences=2):
    """
    Concatenate headline + first n_sentences of articleBody.
    FinBERT input for Kotekar dataset (polarity signal is front-loaded).
    """
    headline = str(row['headline']).strip()
    body = str(row['articleBody']).strip()
    sentences = re.split(r'(?<=[.!?])\s+', body)
    first_n = ' '.join(sentences[:n_sentences])
    return f"{headline}. {first_n}"


# ─────────────────────────────────────────────────────────────────────────────
# Inference functions
# ─────────────────────────────────────────────────────────────────────────────

def get_finbert_polarity(texts, tokenizer, model, device,
                         batch_size=32, max_length=512):
    """
    Compute continuous polarity score in [-1, 1] for each text.
    polarity = P(positive) - P(negative)
    ProsusAI/finbert label order: positive=0, negative=1, neutral=2

    Args:
        texts:      list of strings
        tokenizer:  loaded FinBERT tokenizer
        model:      loaded FinBERT model (on device, eval mode)
        device:     'cuda' or 'cpu'
        batch_size: int
        max_length: int (max tokens, default 512)

    Returns:
        list of float, length == len(texts)
    """
    polarities = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])
        inputs = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        # positive=0, negative=1
        polarity = (probs[:, 0] - probs[:, 1]).cpu().numpy()
        polarities.extend(polarity.tolist())
        if i % 500 == 0:
            print(f"  FinBERT: {i}/{len(texts)}")
    return polarities


def get_subjectivity(texts, tokenizer, model, device,
                     hypothesis=None, batch_size=16, max_length=512):
    """
    Compute subjectivity score in [0, 1] via NLI entailment probability.
    Uses cross-encoder/nli-deberta-v3-small.
    Higher score = more subjective / opinionated text.

    mDeBERTa NLI label order (3-class): 0=contradiction, 1=neutral, 2=entailment

    Args:
        texts:      list of strings (full articleBody, truncated to max_length tokens)
        tokenizer:  loaded mDeBERTa tokenizer
        model:      loaded mDeBERTa model (on device, eval mode)
        device:     'cuda' or 'cpu'
        hypothesis: string (default from DECISIONS.md)
        batch_size: int
        max_length: int

    Returns:
        list of float, length == len(texts)
    """
    if hypothesis is None:
        hypothesis = "This text expresses a personal opinion or subjective view."

    scores = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])
        inputs = tokenizer(
            batch,
            [hypothesis] * len(batch),
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        subjectivity = probs[:, 2].cpu().numpy()  # index 2 = entailment
        scores.extend(subjectivity.tolist())
        if i % 200 == 0:
            print(f"  mDeBERTa: {i}/{len(texts)}")
    return scores
