"""
N-Gram Language Model Implementation
This module implements a Trigram (N=3) language model for text generation.
"""

import re
import random
from collections import defaultdict
from typing import List, Dict, Tuple


class TrigramModel:
    """
    A Trigram Language Model that learns patterns from text and generates new text.
    
    The model uses a nested dictionary structure to store trigram counts:
    counts[word1][word2][word3] = count
    
    Special tokens:
    - <START>: Marks the beginning of a sentence
    - <END>: Marks the end of a sentence
    - <UNK>: Represents unknown/rare words
    """
    
    def __init__(self, min_word_count: int = 2):
        """
        Initialize the Trigram Model.
        
        Args:
            min_word_count: Minimum frequency for a word to be included in vocabulary.
                           Words appearing less than this will be replaced with <UNK>.
        """
        self.min_word_count = min_word_count
        self.vocab = set()
        self.word_counts = defaultdict(int)
        
        # Nested dictionary: trigram_counts[w1][w2][w3] = count
        self.trigram_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        # Bigram counts for context: bigram_counts[w1][w2] = count
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        
        # Special tokens
        self.START = "<START>"
        self.END = "<END>"
        self.UNK = "<UNK>"
        
    def clean_text(self, text: str) -> List[str]:
        """
        Clean and tokenize the input text.
        
        Args:
            text: Raw input text
            
        Returns:
            List of cleaned sentences (each sentence is a list of words)
        """
        # Convert to lowercase
        text = text.lower()
        
        # Split into sentences (simple approach using common sentence endings)
        sentences = re.split(r'[.!?]+', text)
        
        cleaned_sentences = []
        for sentence in sentences:
            # Remove special characters, keep only letters and spaces
            sentence = re.sub(r'[^a-z\s]', '', sentence)
            
            # Tokenize by whitespace
            words = sentence.split()
            
            # Only keep non-empty sentences
            if words:
                cleaned_sentences.append(words)
        
        return cleaned_sentences
    
    def build_vocabulary(self, sentences: List[List[str]]) -> None:
        """
        Build vocabulary from sentences, keeping only words that appear
        at least min_word_count times.
        
        Args:
            sentences: List of sentences (each sentence is a list of words)
        """
        # Count word frequencies
        for sentence in sentences:
            for word in sentence:
                self.word_counts[word] += 1
        
        # Build vocabulary with words meeting minimum count threshold
        self.vocab = {word for word, count in self.word_counts.items() 
                     if count >= self.min_word_count}
        
        # Always include special tokens
        self.vocab.add(self.START)
        self.vocab.add(self.END)
        self.vocab.add(self.UNK)
    
    def replace_unknown_words(self, sentence: List[str]) -> List[str]:
        """
        Replace words not in vocabulary with <UNK> token.
        
        Args:
            sentence: List of words
            
        Returns:
            Sentence with unknown words replaced
        """
        return [word if word in self.vocab else self.UNK for word in sentence]
    
    def pad_sentence(self, sentence: List[str]) -> List[str]:
        """
        Add START and END tokens to sentence for proper trigram extraction.
        
        Args:
            sentence: List of words
            
        Returns:
            Padded sentence
        """
        # Add two START tokens (for trigram context) and one END token
        return [self.START, self.START] + sentence + [self.END]
    
    def extract_trigrams(self, sentence: List[str]) -> List[Tuple[str, str, str]]:
        """
        Extract all trigrams from a sentence.
        
        Args:
            sentence: Padded sentence with START and END tokens
            
        Returns:
            List of trigrams (3-tuples of words)
        """
        trigrams = []
        for i in range(len(sentence) - 2):
            trigrams.append((sentence[i], sentence[i+1], sentence[i+2]))
        return trigrams
    
    def fit(self, text: str) -> None:
        """
        Train the trigram model on the given text.
        
        Args:
            text: Raw training text
        """
        print("Cleaning text...")
        sentences = self.clean_text(text)
        print(f"Found {len(sentences)} sentences")
        
        print("Building vocabulary...")
        self.build_vocabulary(sentences)
        print(f"Vocabulary size: {len(self.vocab)} words")
        
        print("Counting trigrams...")
        for sentence in sentences:
            # Replace unknown words
            sentence = self.replace_unknown_words(sentence)
            
            # Pad with START and END tokens
            padded = self.pad_sentence(sentence)
            
            # Extract and count trigrams
            trigrams = self.extract_trigrams(padded)
            for w1, w2, w3 in trigrams:
                self.trigram_counts[w1][w2][w3] += 1
                self.bigram_counts[w1][w2] += 1
        
        print(f"Model trained on {sum(sum(sum(counts.values()) for counts in bigram.values()) for bigram in self.trigram_counts.values())} trigrams")
    
    def get_next_word_probabilities(self, w1: str, w2: str) -> Dict[str, float]:
        """
        Calculate probability distribution for the next word given two previous words.
        
        Args:
            w1: First context word
            w2: Second context word
            
        Returns:
            Dictionary mapping words to their probabilities
        """
        # Get counts for all possible next words
        possible_words = self.trigram_counts[w1][w2]
        
        if not possible_words:
            # No trigram found, return uniform distribution over vocabulary
            # (or could use bigram/unigram fallback)
            return {word: 1.0 / len(self.vocab) for word in self.vocab}
        
        # Calculate total count for normalization
        total = sum(possible_words.values())
        
        # Convert counts to probabilities
        probabilities = {word: count / total for word, count in possible_words.items()}
        
        return probabilities
    
    def sample_next_word(self, w1: str, w2: str) -> str:
        """
        Probabilistically sample the next word given two previous words.
        
        Args:
            w1: First context word
            w2: Second context word
            
        Returns:
            Sampled next word
        """
        probabilities = self.get_next_word_probabilities(w1, w2)
        
        # Get words and their probabilities
        words = list(probabilities.keys())
        probs = list(probabilities.values())
        
        # Sample using the probability distribution
        next_word = random.choices(words, weights=probs, k=1)[0]
        
        return next_word
    
    def generate(self, max_length: int = 50, seed: str = None) -> str:
        """
        Generate new text using the trained trigram model.
        
        Args:
            max_length: Maximum number of words to generate
            seed: Optional seed text to start generation (will be cleaned)
            
        Returns:
            Generated text string
        """
        # Start with two START tokens
        if seed:
            # Clean and use seed
            seed_words = self.clean_text(seed)
            if seed_words and seed_words[0]:
                context = self.replace_unknown_words(seed_words[0][-2:])
                if len(context) == 1:
                    context = [self.START] + context
                elif len(context) == 0:
                    context = [self.START, self.START]
                generated = context.copy()
            else:
                context = [self.START, self.START]
                generated = []
        else:
            context = [self.START, self.START]
            generated = []
        
        # Generate words
        for _ in range(max_length):
            w1, w2 = context[-2], context[-1]
            
            # Sample next word
            next_word = self.sample_next_word(w1, w2)
            
            # Stop if we hit END token
            if next_word == self.END:
                break
            
            # Don't add START tokens to output
            if next_word != self.START:
                generated.append(next_word)
            
            # Update context
            context.append(next_word)
        
        # Join words into a sentence
        return ' '.join(generated)
    
    def perplexity(self, text: str) -> float:
        """
        Calculate perplexity of the model on given text.
        Lower perplexity indicates better model performance.
        
        Args:
            text: Test text
            
        Returns:
            Perplexity score
        """
        sentences = self.clean_text(text)
        log_prob_sum = 0
        word_count = 0
        
        for sentence in sentences:
            sentence = self.replace_unknown_words(sentence)
            padded = self.pad_sentence(sentence)
            trigrams = self.extract_trigrams(padded)
            
            for w1, w2, w3 in trigrams:
                probs = self.get_next_word_probabilities(w1, w2)
                prob = probs.get(w3, 1e-10)  # Small probability for unseen words
                log_prob_sum += -1 * (prob ** 0.5)  # Simplified calculation
                word_count += 1
        
        # Calculate perplexity
        if word_count == 0:
            return float('inf')
        
        avg_log_prob = log_prob_sum / word_count
        perplexity = 2 ** avg_log_prob
        
        return perplexity


# Example usage
if __name__ == "__main__":
    # Example training text
    sample_text = """
    Alice was beginning to get very tired of sitting by her sister on the bank.
    She had nothing to do. Once or twice she had peeped into the book her sister was reading.
    But it had no pictures or conversations in it. And what is the use of a book without pictures?
    Alice was considering in her own mind whether the pleasure of making a daisy chain would be worth the trouble.
    """
    
    # Create and train model
    model = TrigramModel(min_word_count=1)
    model.fit(sample_text)
    
    # Generate text
    print("\n" + "="*50)
    print("Generated Text:")
    print("="*50)
    for i in range(3):
        generated = model.generate(max_length=30)
        print(f"{i+1}. {generated}")
    
    print("\n" + "="*50)
    print("Generated with seed 'Alice was':")
    print("="*50)
    generated = model.generate(max_length=30, seed="Alice was")
    print(generated)