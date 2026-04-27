"""
guardianai.utils.vectorizer
============================
Shared DualVectorizer class used by both the training pipeline and
the PromptInspector agent. Must live in a stable importable module
so joblib can deserialise the saved vectorizer correctly.
"""

from scipy.sparse import hstack


class DualVectorizer:
    """
    Wraps two TfidfVectorizers (word n-gram + char_wb n-gram) into a
    single .transform() API compatible with the existing agent interface.

    Saved as prompt_vectorizer.joblib by the training pipeline.
    Loaded and called by PromptInspector._ml_score().
    """
    def __init__(self, word_vec, char_vec):
        self.word_vec = word_vec
        self.char_vec = char_vec

    def transform(self, texts):
        return hstack([
            self.word_vec.transform(texts),
            self.char_vec.transform(texts),
        ])