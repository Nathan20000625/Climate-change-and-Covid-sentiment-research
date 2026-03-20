#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reddit COVID-19 and climate-change sentiment analysis
Use NRC Lexicon to analyze sentiment from the `body` column
"""

import pandas as pd
import numpy as np
import os
import sys
try:
    from NRCLex import NRCLex
except ImportError:
    try:
        from nrclex import NRCLex
    except ImportError:
        print("Warning: failed to import NRCLex. Please install it: pip install NRCLex")
        NRCLex = None
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def _ensure_nlp_corpora_available():
    """
    Ensure required NLTK/TextBlob corpora are available.
    - NRCLex depends on NLTK resources (e.g., tokenizers); some environments may also
      indirectly trigger TextBlob corpora checks.
    - Download missing resources quietly to avoid interactive prompts.
    """
    try:
        import nltk
        from nltk.data import find as nltk_find

        # Required NLTK resources (download if missing)
        required = [
            ("tokenizers/punkt", "punkt"),
            ("corpora/wordnet", "wordnet"),
            ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
            ("corpora/stopwords", "stopwords"),
        ]
        
        print("Checking NLTK corpora...")
        for resource_path, download_name in required:
            try:
                nltk_find(resource_path)
            except LookupError:
                print(f"Downloading NLTK resource: {download_name}")
                nltk.download(download_name, quiet=True)

    except Exception as e:
        print(f"Note: issue while checking/downloading NLTK resources: {e}")

    # Proactively download TextBlob corpora
    try:
        import textblob
        print("Downloading TextBlob corpora...")
        # Use subprocess to run TextBlob's corpora download command
        import subprocess
        import sys
        
        # Run: python -m textblob.download_corpora
        result = subprocess.run([
            sys.executable, "-m", "textblob.download_corpora"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("TextBlob corpora download succeeded!")
        else:
            print(f"TextBlob corpora download failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("TextBlob corpora download timed out. Please try again later.")
    except Exception as e:
        print(f"Issue while downloading TextBlob corpora: {e}")
        print("You can run manually: python -m textblob.download_corpora")

class SentimentAnalyzer:
    def __init__(self):
        """
        Initialize sentiment analyzer using NRC Lexicon.
        Produces 8 Plutchik emotions + positive/negative labels.
        """
        if NRCLex is None:
            raise ImportError("NRCLex is not installed. Please run: pip install NRCLex")
            
        print("Initializing NRC Lexicon sentiment analyzer...")

        # Ensure required corpora are available
        print("Preparing required corpora for sentiment analysis...")
        _ensure_nlp_corpora_available()
        print("Corpora ready!")
        
        # Emotion labels supported by NRC Lexicon
        self.emotion_labels = {
            'joy': 'joy',
            'sadness': 'sadness',
            'anger': 'anger',
            'fear': 'fear',
            'surprise': 'surprise',
            'disgust': 'disgust',
            'trust': 'trust',
            'anticipation': 'anticipation',
            'positive': 'positive',
            'negative': 'negative'
        }
        
        print("NRC Lexicon initialized! Using 8 Plutchik emotions + positive/negative labels.")
    
    def analyze_sentiment(self, text):
        """
        Analyze a single text with 8 Plutchik emotions and positive/negative scores.
        
        Args:
            text (str): text to analyze
            
        Returns:
            dict: results containing emotion, confidence, and raw emotion scores
        """
        # Default result (8 Plutchik emotions + positive/negative)
        default_result = {'emotion': 'neutral', 'confidence': 0.0}
        for emotion in self.emotion_labels.keys():
            default_result[emotion] = 0.0
        
        if pd.isna(text) or text == "":
            return default_result
        
        try:
            # NRC Lexicon sentiment analysis
            nrc_analyzer = NRCLex(text)
            
            # Collect raw emotion scores (8 Plutchik emotions + positive/negative)
            emotion_scores = {}
            for emotion in self.emotion_labels.keys():
                emotion_scores[emotion] = nrc_analyzer.raw_emotion_scores.get(emotion, 0)
            
            # Select the best emotion by raw score
            if sum(emotion_scores.values()) > 0:
                best_emotion = max(emotion_scores, key=emotion_scores.get)
                # Confidence = max_score / total_score
                confidence = emotion_scores[best_emotion] / sum(emotion_scores.values())
            else:
                best_emotion = 'neutral'
                confidence = 0.0
            
            return {
                'emotion': best_emotion,
                'confidence': confidence,
                **emotion_scores  # raw scores
            }
            
        except Exception as e:
            print(f"Error analyzing text: {e}")
            return default_result
    
    def analyze_batch(self, texts, batch_size=32):
        """
        Analyze sentiment for a list of texts.
        
        Args:
            texts (list): list of texts
            batch_size (int): batch size (kept for API compatibility; NRCLex itself is per-text)
            
        Returns:
            list: list of sentiment analysis results
        """
        results = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing sentiment"):
            batch = texts[i:i+batch_size]
            batch_results = []
            
            for text in batch:
                result = self.analyze_sentiment(text)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results

def main():
    """Main entry point."""
    # File paths
    csv_file = r"F:\reddit_covid_climate\fixed\merged.csv"
    output_file = r"F:\reddit_covid_climate\NRC\NRClexicon_results.csv"
    
    print("=== Reddit COVID-19 and climate-change sentiment analysis ===")
    print("Using NRC Lexicon: 8 Plutchik emotions + positive/negative scores")
    
    print("\nStarting sentiment analysis...")
    print(f"Input file: {csv_file}")
    print(f"Output file: {output_file}")
    
    try:
        # If output exists, delete it to avoid appending to old results
        if os.path.exists(output_file):
            print("Existing output file detected. Deleting to regenerate...")
            os.remove(output_file)

        # Initialize analyzer (once)
        analyzer = SentimentAnalyzer()

        # Chunked processing
        chunk_size = 5000
        print(f"Processing in chunks: chunksize={chunk_size} rows")

        header_written = False
        chunk_iter = pd.read_csv(csv_file, chunksize=chunk_size)

        for chunk_index, df in enumerate(chunk_iter, start=1):
            # Column check on first chunk
            if chunk_index == 1 and 'body' not in df.columns:
                print("Error: column 'body' not found in the CSV file.")
                print(f"Available columns: {list(df.columns)}")
                return

            # Prepare text
            body_texts = df['body'].fillna('').astype(str).tolist()

            # Sentiment analysis (internally loops over small batches)
            tqdm_desc = f"Analyzing sentiment (chunk {chunk_index})"
            sentiment_results = analyzer.analyze_batch(body_texts, batch_size=32)

            # Write sentiment result columns
            df['emotion'] = [result['emotion'] for result in sentiment_results]
            df['emotion_confidence'] = [result['confidence'] for result in sentiment_results]
            for emotion in analyzer.emotion_labels.keys():
                column_name = f"{emotion}_score"
                df[column_name] = [result[emotion] for result in sentiment_results]

            # Vectorized word count
            df['word_count'] = df['body'].fillna('').astype(str).str.split().str.len()

            # Append-write current chunk
            df.to_csv(
                output_file,
                index=False,
                encoding='utf-8',
                mode='a',
                header=not header_written
            )
            header_written = True

            # Progress message
            print(f"Finished chunk {chunk_index}; wrote {len(df)} rows")

        print(f"\nSentiment analysis complete! Results saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: file not found: {csv_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
