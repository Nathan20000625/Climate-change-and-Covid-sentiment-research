import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def analyze_emotion_scores():
    """
    Analyze NRC Lexicon results and compute daily mean emotion frequencies.
    """
    print("Reading CSV in chunks...")
    
    # Read in chunks to save memory
    chunk_size = 100000  # 100k rows per chunk
    chunks = []
    batch_count = 0  # number of raw chunks currently buffered
    
    # Columns to load (exclude `body` to save memory)
    usecols = [
        'subreddit', 'author', 'created', 'score', 'link', 'climate', 'covid', 
        'emotion', 'emotion_confidence', 'joy_score', 'sadness_score', 'anger_score', 
        'fear_score', 'surprise_score', 'disgust_score', 'trust_score', 
        'anticipation_score', 'positive_score', 'negative_score', 'word_count'
    ]
    
    for i, chunk in enumerate(pd.read_csv(r'E:\reddit_covid_climate\NRC\NRClexicon_results.csv', 
                                         chunksize=chunk_size, usecols=usecols)):
        print(f"Processing chunk {i+1}...")
        
        # Preprocess each chunk
        # 1) Keep rows with word_count > 15
        chunk = chunk[chunk['word_count'] > 15]
        
        # 2) Exclude bot-like authors
        chunk = chunk[~chunk['author'].str.contains('bot|_bot|-bot', case=False, na=False)]
        
        # 3) Parse `created` as datetime (mixed formats)
        chunk['created'] = pd.to_datetime(chunk['created'], format='mixed', errors='coerce')
        chunk['date'] = chunk['created'].dt.date
        
        # 4) Create groups
        def create_group(row):
            if row['climate'] == 1 and row['covid'] == 1:
                return 'both'
            elif row['climate'] == 1:
                return 'climate'
            elif row['covid'] == 1:
                return 'covid'
            else:
                return 'other'
        
        chunk['group'] = chunk.apply(create_group, axis=1)
        
        # Keep meaningful groups only
        chunk = chunk[chunk['group'].isin(['climate', 'covid', 'both'])]
        
        chunks.append(chunk)
        batch_count += 1
        
        # Merge every 5 chunks to avoid memory growth
        if batch_count >= 5:
            print(f"Merging previous {batch_count} chunks...")
            df_temp = pd.concat(chunks, ignore_index=True)
            chunks = [df_temp]
            batch_count = 0  # reset counter
    
    # Merge all chunks
    print("Merging all chunks...")
    df = pd.concat(chunks, ignore_index=True)
    
    print(f"Done. Final row count: {len(df)}")
    print(f"Column count: {len(df.columns)}")
    
    # Preprocessing done; show quick stats
    print("\nPreprocessing complete.")
    
    # Group counts
    print("\nGroup counts:")
    print(df['group'].value_counts())
    
    # Emotion score columns
    emotion_columns = [
        'joy_score', 'sadness_score', 'anger_score', 'fear_score',
        'surprise_score', 'disgust_score', 'trust_score', 'anticipation_score',
        'positive_score', 'negative_score'
    ]
    
    print(f"\nEmotion score columns: {emotion_columns}")
    
    # Compute emotion frequencies (score / word_count)
    print("\nComputing emotion frequencies...")
    for emotion in emotion_columns:
        df[f'{emotion}_freq'] = df[emotion] / df['word_count']
    
    # Emotion frequency columns
    emotion_freq_columns = [f'{emotion}_freq' for emotion in emotion_columns]
    print(f"Emotion frequency columns: {emotion_freq_columns}")
    
    # Daily mean emotion frequencies by (date, group)
    print("\nComputing daily mean emotion frequencies...")
    
    daily_stats = df.groupby(['date', 'group'])[emotion_freq_columns].mean().reset_index()
    
    # Add sample counts
    daily_counts = df.groupby(['date', 'group']).size().reset_index(name='sample_count')
    daily_stats = daily_stats.merge(daily_counts, on=['date', 'group'])
    
    # Reshape: one row per date, group-specific emotion columns
    print("\nReshaping output...")
    
    # Pivot with group as columns
    pivot_data = daily_stats.pivot_table(
        index='date', 
        columns='group', 
        values=emotion_freq_columns, 
        fill_value=np.nan
    )
    
    # Flatten multi-index columns
    pivot_data.columns = [f"{group}_{emotion}" for emotion, group in pivot_data.columns]
    
    # Reset index to keep `date` as a column
    pivot_data = pivot_data.reset_index()
    
    # Reorder columns: date, then for each emotion place climate/covid/both together
    column_order = ['date']
    for emotion in emotion_freq_columns:
        for group in ['climate', 'covid', 'both']:
            column_order.append(f"{group}_{emotion}")
    
    # Keep existing columns only
    existing_columns = [col for col in column_order if col in pivot_data.columns]
    pivot_data = pivot_data[existing_columns]
    
    # Sort by date
    pivot_data = pivot_data.sort_values('date')
    
    # Save reshaped output
    pivot_data.to_csv('NRC_daily_emotion_frequency_pivot.csv', index=False)
    print("\nSaved reshaped output to NRC_daily_emotion_frequency_pivot.csv")
    print(f"Shape: {pivot_data.shape[0]} rows x {pivot_data.shape[1]} columns")
    
    # Show columns
    print("\nOutput columns:")
    for i, col in enumerate(pivot_data.columns):
        print(f"{i+1:2d}. {col}")
    
    # Save long-format grouped output (optional)
    daily_stats.to_csv('NRC_daily_emotion_frequency_by_group.csv', index=False)
    print("\nSaved grouped output to NRC_daily_emotion_frequency_by_group.csv")
    
    # Summary
    print("\n=== Summary ===")
    print(f"Total days: {len(pivot_data)}")
    print(f"Date range: {pivot_data['date'].min()} to {pivot_data['date'].max()}")
    print(f"Total columns: {len(pivot_data.columns)} (1 date column + {len(pivot_data.columns)-1} emotion columns)")
    
    # Group availability
    print("\n=== Group coverage ===")
    for group in ['climate', 'covid', 'both']:
        group_cols = [col for col in pivot_data.columns if col.startswith(f"{group}_")]
        if group_cols:
            non_null_count = pivot_data[group_cols].notna().any(axis=1).sum()
            print(f"{group}: {non_null_count} days with data")
    
    return pivot_data

if __name__ == "__main__":
    try:
        result = analyze_emotion_scores()
        print("\nAnalysis completed!")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
