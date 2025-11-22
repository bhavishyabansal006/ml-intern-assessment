"""
Training script for the Trigram Language Model
"""

import argparse
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ngram_model import TrigramModel


def load_text(filepath: str) -> str:
    """
    Load text from a file.
    
    Args:
        filepath: Path to text file
        
    Returns:
        Text content
    """
    print(f"Loading text from: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"✓ Loaded {len(text):,} characters")
    return text


def train_model(text: str, min_word_count: int = 2) -> TrigramModel:
    """
    Train a trigram model on the given text.
    
    Args:
        text: Training text
        min_word_count: Minimum word frequency threshold
        
    Returns:
        Trained TrigramModel
    """
    print(f"\n{'='*60}")
    print("Training Trigram Model")
    print(f"{'='*60}")
    
    model = TrigramModel(min_word_count=min_word_count)
    model.fit(text)
    
    print(f"\n✓ Training complete!")
    print(f"  - Vocabulary size: {len(model.vocab)}")
    print(f"  - Unique trigrams: {sum(len(bigram) for w1 in model.trigram_counts.values() for bigram in w1.values())}")
    
    return model


def generate_samples(model: TrigramModel, num_samples: int = 5, max_length: int = 50):
    """
    Generate sample text from the trained model.
    
    Args:
        model: Trained TrigramModel
        num_samples: Number of samples to generate
        max_length: Maximum length of each sample
    """
    print(f"\n{'='*60}")
    print(f"Generated Text Samples (max {max_length} words each)")
    print(f"{'='*60}\n")
    
    for i in range(num_samples):
        generated = model.generate(max_length=max_length)
        print(f"Sample {i+1}:")
        print(f"  {generated}\n")


def interactive_mode(model: TrigramModel):
    """
    Interactive text generation mode.
    
    Args:
        model: Trained TrigramModel
    """
    print(f"\n{'='*60}")
    print("Interactive Generation Mode")
    print(f"{'='*60}")
    print("Commands:")
    print("  - Type a seed phrase to generate text from it")
    print("  - Type 'random' to generate random text")
    print("  - Type 'quit' to exit")
    print(f"{'='*60}\n")
    
    while True:
        try:
            user_input = input("Enter seed (or 'random'/'quit'): ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            if user_input.lower() == 'random' or not user_input:
                generated = model.generate(max_length=50)
            else:
                generated = model.generate(max_length=50, seed=user_input)
            
            print(f"\nGenerated: {generated}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main training and generation pipeline"""
    parser = argparse.ArgumentParser(description='Train a Trigram Language Model')
    parser.add_argument(
        '--data',
        type=str,
        default='data/cleaned_pride_and_prejudice.txt',
        help='Path to training text file'
    )
    parser.add_argument(
        '--min-count',
        type=int,
        default=2,
        help='Minimum word count for vocabulary'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=5,
        help='Number of sample texts to generate'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=50,
        help='Maximum length of generated text'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Enable interactive generation mode'
    )
    
    args = parser.parse_args()
    
    try:
        # Load data
        text = load_text(args.data)
        
        # Train model
        model = train_model(text, min_word_count=args.min_count)
        
        # Generate samples
        generate_samples(model, num_samples=args.samples, max_length=args.max_length)
        
        # Interactive mode
        if args.interactive:
            interactive_mode(model)
        else:
            print("\nTip: Run with --interactive flag for interactive generation!")
        
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease run the data extraction script first:")
        print("  python extract_data.py")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()