"""
Data extraction and cleaning script for Project Gutenberg books
This script downloads and preprocesses text data for training the trigram model.
"""

import urllib.request
import re
import os


class GutenbergDownloader:
    """Download and clean books from Project Gutenberg"""
    
    BOOKS = {
        'alice': {
            'url': 'https://www.gutenberg.org/files/11/11-0.txt',
            'title': 'Alice\'s Adventures in Wonderland',
            'filename': 'alice_in_wonderland.txt'
        },
        'pride': {
            'url': 'https://www.gutenberg.org/files/1342/1342-0.txt',
            'title': 'Pride and Prejudice',
            'filename': 'pride_and_prejudice.txt'
        },
        'frankenstein': {
            'url': 'https://www.gutenberg.org/files/84/84-0.txt',
            'title': 'Frankenstein',
            'filename': 'frankenstein.txt'
        },
        'tale_two_cities': {
            'url': 'https://www.gutenberg.org/files/98/98-0.txt',
            'title': 'A Tale of Two Cities',
            'filename': 'tale_of_two_cities.txt'
        }
    }
    
    def __init__(self, data_dir='data'):
        """
        Initialize the downloader.
        
        Args:
            data_dir: Directory to save downloaded books
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download_book(self, book_key: str) -> str:
        """
        Download a book from Project Gutenberg.
        
        Args:
            book_key: Key from BOOKS dictionary ('alice', 'pride', etc.)
            
        Returns:
            Path to downloaded file
        """
        if book_key not in self.BOOKS:
            raise ValueError(f"Unknown book: {book_key}. Choose from {list(self.BOOKS.keys())}")
        
        book_info = self.BOOKS[book_key]
        url = book_info['url']
        filename = os.path.join(self.data_dir, book_info['filename'])
        
        print(f"Downloading '{book_info['title']}'...")
        print(f"URL: {url}")
        
        try:
            # Download the file
            urllib.request.urlretrieve(url, filename)
            print(f"✓ Downloaded to: {filename}")
            return filename
        except Exception as e:
            print(f"✗ Error downloading: {e}")
            raise
    
    def clean_gutenberg_text(self, filepath: str) -> str:
        """
        Clean Project Gutenberg text by removing headers, footers, and formatting.
        
        Args:
            filepath: Path to the downloaded text file
            
        Returns:
            Cleaned text content
        """
        print(f"\nCleaning text from: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Remove Project Gutenberg header (usually before "*** START OF")
        start_marker = re.search(r'\*\*\* START OF (THE|THIS) PROJECT GUTENBERG', text, re.IGNORECASE)
        if start_marker:
            text = text[start_marker.end():]
        
        # Remove Project Gutenberg footer (usually after "*** END OF")
        end_marker = re.search(r'\*\*\* END OF (THE|THIS) PROJECT GUTENBERG', text, re.IGNORECASE)
        if end_marker:
            text = text[:end_marker.start()]
        
        # Remove chapter headers like "CHAPTER I", "CHAPTER 1", etc.
        text = re.sub(r'CHAPTER [IVXLC\d]+\.?\s*\n', '', text, flags=re.IGNORECASE)
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove lines that are all caps (often headers)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Keep line if it's not all caps or is short
            if not stripped or not stripped.isupper() or len(stripped) < 10:
                cleaned_lines.append(line)
            elif any(char.islower() for char in stripped):
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Final whitespace cleanup
        text = text.strip()
        
        print(f"✓ Text cleaned ({len(text)} characters)")
        return text
    
    def save_cleaned_text(self, text: str, output_filename: str) -> str:
        """
        Save cleaned text to a file.
        
        Args:
            text: Cleaned text content
            output_filename: Name for the output file
            
        Returns:
            Path to saved file
        """
        output_path = os.path.join(self.data_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"✓ Saved cleaned text to: {output_path}")
        return output_path
    
    def download_and_clean(self, book_key: str) -> str:
        """
        Download and clean a book in one step.
        
        Args:
            book_key: Key from BOOKS dictionary
            
        Returns:
            Path to cleaned text file
        """
        # Download
        raw_filepath = self.download_book(book_key)
        
        # Clean
        cleaned_text = self.clean_gutenberg_text(raw_filepath)
        
        # Save cleaned version
        cleaned_filename = f"cleaned_{self.BOOKS[book_key]['filename']}"
        cleaned_filepath = self.save_cleaned_text(cleaned_text, cleaned_filename)
        
        # Print statistics
        print(f"\n{'='*60}")
        print(f"Summary for '{self.BOOKS[book_key]['title']}':")
        print(f"  - Characters: {len(cleaned_text):,}")
        print(f"  - Words (approx): {len(cleaned_text.split()):,}")
        print(f"  - Location: {cleaned_filepath}")
        print(f"{'='*60}\n")
        
        return cleaned_filepath


def main():
    """Main function to demonstrate usage"""
    print("="*60)
    print("Project Gutenberg Book Downloader and Cleaner")
    print("="*60)
    
    downloader = GutenbergDownloader(data_dir='data')
    
    print("\nAvailable books:")
    for i, (key, info) in enumerate(downloader.BOOKS.items(), 1):
        print(f"  {i}. {info['title']} (key: '{key}')")
    
    print("\n" + "="*60)
    
    # Download and clean Pride and Prejudice (recommended)
    print("\nDownloading recommended book: Pride and Prejudice")
    print("="*60)
    
    try:
        cleaned_file = downloader.download_and_clean('pride')
        print(f"\n✓ SUCCESS! Use this file for training: {cleaned_file}")
        
        print("\nTo train your model, run:")
        print(f"  python train_model.py --data {cleaned_file}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nIf download fails, you can manually:")
        print("1. Go to https://www.gutenberg.org/")
        print("2. Search for 'Pride and Prejudice'")
        print("3. Download as Plain Text UTF-8")
        print("4. Save to data/pride_and_prejudice.txt")


if __name__ == "__main__":
    main()