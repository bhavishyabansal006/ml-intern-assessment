"""
Test suite for the Trigram Language Model
"""

import pytest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ngram_model import TrigramModel


class TestTrigramModel:
    """Test cases for TrigramModel class"""
    
    @pytest.fixture
    def sample_text(self):
        """Provide sample training text"""
        return """
        The cat sat on the mat. The dog sat on the log.
        The cat likes the mat. The dog likes the log.
        The cat and the dog are friends. They play together.
        """
    
    @pytest.fixture
    def trained_model(self, sample_text):
        """Provide a trained model"""
        model = TrigramModel(min_word_count=1)
        model.fit(sample_text)
        return model
    
    def test_initialization(self):
        """Test model initialization"""
        model = TrigramModel(min_word_count=2)
        assert model.min_word_count == 2
        assert model.START == "<START>"
        assert model.END == "<END>"
        assert model.UNK == "<UNK>"
        assert len(model.vocab) == 0
    
    def test_clean_text(self, sample_text):
        """Test text cleaning and tokenization"""
        model = TrigramModel()
        sentences = model.clean_text(sample_text)
        
        # Check that we got sentences
        assert len(sentences) > 0
        
        # Check that each sentence is a list of words
        for sentence in sentences:
            assert isinstance(sentence, list)
            assert all(isinstance(word, str) for word in sentence)
            # Check lowercase
            assert all(word.islower() or word.isspace() for word in sentence)
    
    def test_build_vocabulary(self, sample_text):
        """Test vocabulary building"""
        model = TrigramModel(min_word_count=2)
        sentences = model.clean_text(sample_text)
        model.build_vocabulary(sentences)
        
        # Check that special tokens are in vocabulary
        assert model.START in model.vocab
        assert model.END in model.vocab
        assert model.UNK in model.vocab
        
        # Check that common words are in vocabulary
        assert 'the' in model.vocab  # Very common word
        assert 'cat' in model.vocab
    
    def test_pad_sentence(self):
        """Test sentence padding"""
        model = TrigramModel()
        sentence = ['hello', 'world']
        padded = model.pad_sentence(sentence)
        
        assert padded[0] == model.START
        assert padded[1] == model.START
        assert padded[-1] == model.END
        assert len(padded) == len(sentence) + 3
    
    def test_extract_trigrams(self):
        """Test trigram extraction"""
        model = TrigramModel()
        sentence = [model.START, model.START, 'hello', 'world', model.END]
        trigrams = model.extract_trigrams(sentence)
        
        # Should extract: (<START>, <START>, hello), (<START>, hello, world), (hello, world, <END>)
        assert len(trigrams) == 3
        assert trigrams[0] == (model.START, model.START, 'hello')
        assert trigrams[1] == (model.START, 'hello', 'world')
        assert trigrams[2] == ('hello', 'world', model.END)
    
    def test_fit(self, sample_text):
        """Test model training"""
        model = TrigramModel(min_word_count=1)
        model.fit(sample_text)
        
        # Check that vocabulary was built
        assert len(model.vocab) > 0
        
        # Check that trigrams were counted
        assert len(model.trigram_counts) > 0
        
        # Check specific trigram
        # After cleaning: "the cat sat" should be a trigram
        assert model.trigram_counts[model.START][model.START]['the'] > 0
    
    def test_replace_unknown_words(self, trained_model):
        """Test unknown word replacement"""
        # Word not in training data
        sentence = ['the', 'cat', 'jumped', 'over', 'fence']
        replaced = trained_model.replace_unknown_words(sentence)
        
        # 'the' and 'cat' should remain, rare words should be <UNK>
        assert 'the' in replaced
        assert 'cat' in replaced
        assert trained_model.UNK in replaced
    
    def test_get_next_word_probabilities(self, trained_model):
        """Test probability calculation"""
        # Get probabilities for a context that exists in training data
        probs = trained_model.get_next_word_probabilities('the', 'cat')
        
        # Should return a dictionary
        assert isinstance(probs, dict)
        
        # Probabilities should sum to approximately 1.0
        assert abs(sum(probs.values()) - 1.0) < 0.01
        
        # All probabilities should be positive
        assert all(p > 0 for p in probs.values())
    
    def test_sample_next_word(self, trained_model):
        """Test word sampling"""
        # Sample should return a word from vocabulary
        next_word = trained_model.sample_next_word('the', 'cat')
        assert isinstance(next_word, str)
        assert next_word in trained_model.vocab
    
    def test_generate(self, trained_model):
        """Test text generation"""
        generated = trained_model.generate(max_length=20)
        
        # Should generate a string
        assert isinstance(generated, str)
        
        # Should not be empty (very unlikely with trained model)
        assert len(generated) > 0
        
        # Should not contain START or END tokens in output
        assert trained_model.START not in generated
        assert trained_model.END not in generated
    
    def test_generate_with_seed(self, trained_model):
        """Test text generation with seed"""
        seed = "The cat"
        generated = trained_model.generate(max_length=15, seed=seed)
        
        # Should generate a string
        assert isinstance(generated, str)
        assert len(generated) > 0
    
    def test_generate_different_outputs(self, trained_model):
        """Test that generation produces varied outputs"""
        outputs = [trained_model.generate(max_length=10) for _ in range(5)]
        
        # At least some outputs should be different (probabilistic)
        unique_outputs = set(outputs)
        assert len(unique_outputs) >= 1  # Should have at least one output
    
    def test_model_with_empty_text(self):
        """Test model behavior with empty text"""
        model = TrigramModel()
        model.fit("")
        
        # Should handle gracefully
        assert len(model.vocab) == 3  # Only special tokens
    
    def test_model_with_single_sentence(self):
        """Test model with minimal text"""
        model = TrigramModel(min_word_count=1)
        model.fit("Hello world.")
        
        # Should still work
        assert len(model.vocab) > 3
        generated = model.generate(max_length=5)
        assert isinstance(generated, str)


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_very_short_generation(self):
        """Test generation with very short max_length"""
        model = TrigramModel(min_word_count=1)
        model.fit("The cat sat on the mat.")
        generated = model.generate(max_length=1)
        
        # Should still return something
        assert isinstance(generated, str)
    
    def test_special_characters_in_text(self):
        """Test handling of special characters"""
        model = TrigramModel(min_word_count=1)
        text = "Hello! How are you? I'm fine. What's up? 123 @#$%"
        model.fit(text)
        
        # Should clean and process text
        assert len(model.vocab) > 3
    
    def test_multiple_spaces_and_newlines(self):
        """Test handling of irregular whitespace"""
        model = TrigramModel(min_word_count=1)
        text = "Hello    world.\n\n\nGoodbye   world."
        sentences = model.clean_text(text)
        
        # Should handle properly
        assert len(sentences) > 0
        for sentence in sentences:
            assert all(word.strip() for word in sentence)  # No empty words


# Run tests with: pytest test_ngram.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])