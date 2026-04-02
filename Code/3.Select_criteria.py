import pandas as pd
import os
import re
import hashlib
from datetime import datetime

# Input file paths
input_file = r"E:\reddit_covid_climate\NRC\NRClexicon_results_afterNRC.csv"
output_file = r"E:\reddit_covid_climate\NRC\NRClexicon_results.csv"

def contains_bot_message(text):
    """
    Check whether the text contains a bot message.
    Case-insensitive matching.
    """
    if pd.isna(text) or text == '':
        return False
    
    text_lower = str(text).lower()
    
    # Bot-message keyword list
    bot_phrases = [
        "i am a bot",
        "i'm a bot",
        "this action was performed automatically",
        "this message was posted by a bot.",
        "this comment was left automatically (by a bot).",
        "you can summon this bot any time in"
    ]
    
    # Check whether any bot phrase is present
    for phrase in bot_phrases:
        if phrase in text_lower:
            return True
    
    return False

def is_english(text):
    """
    Heuristically detect whether the text is English.
    Uses a lightweight approach: checks whether the text is mostly ASCII characters
    (letters, numbers, punctuation, etc.). This is faster than language-detection
    libraries for large files.
    """
    if pd.isna(text) or text == '':
        return False
    
    # Convert to string
    text = str(text)
    
    # If the text is too short, skip
    if len(text.strip()) < 3:
        return False
    
    # Compute ASCII ratio; only sample the first 500 chars for speed
    sample_text = text[:500]
    ascii_count = sum(1 for c in sample_text if ord(c) < 128)
    total_count = len(sample_text)
    
    if total_count == 0:
        return False
    
    # If ASCII ratio is over 70%, treat as English (tunable threshold)
    ascii_ratio = ascii_count / total_count
    
    # Additional signal: presence of many English letters
    english_letters = len(re.findall(r'[a-zA-Z]', sample_text))
    letter_ratio = english_letters / total_count if total_count > 0 else 0
    
    # Require both a high ASCII ratio and a sufficient letter ratio
    return ascii_ratio > 0.7 and letter_ratio > 0.3

def is_bot_author(author):
    """
    Detect whether the `author` value matches a bot-like pattern.
    Checks for: -bot, bot-, _bot, bot_ (case-insensitive).
    """
    if pd.isna(author) or author == '':
        return False
    
    author_lower = str(author).lower()
    bot_patterns = ['-bot', 'bot-', '_bot', 'bot_']
    
    for pattern in bot_patterns:
        if pattern in author_lower:
            return True
    
    return False

def is_valid_date(created_date, min_date):
    """
    Check whether `created` is on/after 2020-01-20.
    Uses pandas `to_datetime` to parse multiple date formats.
    """
    if pd.isna(created_date):
        return False
    
    try:
        # Use pandas `to_datetime` to parse dates (supports multiple formats)
        parsed_date = pd.to_datetime(created_date, errors='coerce')
        
        # If parsing fails, return False
        if pd.isna(parsed_date):
            return False
        
        return parsed_date >= min_date
    except:
        return False

print(f"Reading file: {input_file}")
print("The file is large; please wait...")

# Read a small sample first to inspect column names
sample = pd.read_csv(input_file, nrows=5)
print(f"CSV columns: {list(sample.columns)}")

# Check whether required columns exist
required_columns = ['body', 'author', 'word_count', 'created']
missing_columns = [col for col in required_columns if col not in sample.columns]
if missing_columns:
    print(f"Error: the following required columns are missing: {missing_columns}")
    print(f"Available columns: {list(sample.columns)}")
    exit(1)

# Check whether `climate` and `covid` columns exist (for daily stats)
has_climate_covid = 'climate' in sample.columns and 'covid' in sample.columns
if not has_climate_covid:
    print("Warning: 'climate' and/or 'covid' columns not found; daily stats will be skipped.")
    print(f"Available columns: {list(sample.columns)}")

# Minimum date: 2020-01-20
min_date = datetime(2020, 1, 20)

# Memory-friendly approach: process in chunks and deduplicate
chunk_size = 50000  # 50k rows per chunk
# Track hashes instead of full text to save memory and speed up lookups
seen_body_hashes = set()  # hashes of previously-seen `body` values
total_rows = 0
unique_rows = 0
english_rows = 0
bot_body_rows = 0
bot_author_rows = 0
word_count_filtered = 0
date_filtered = 0
first_chunk = True

print("Start chunk processing (bot message + bot author + English + word_count + date + dedup)...")
print("Deduplicating via hashes for better performance and lower memory use...")

# Open output file for writing
with open(output_file, 'w', encoding='utf-8', newline='') as f_out:
    for chunk in pd.read_csv(input_file, chunksize=chunk_size, low_memory=False):
        total_rows += len(chunk)
        
        # Step 1: remove rows whose `body` contains bot messages
        chunk_no_bot_body = chunk[~chunk['body'].apply(contains_bot_message)]
        bot_body_count = len(chunk) - len(chunk_no_bot_body)
        bot_body_rows += bot_body_count
        
        # Step 2: remove rows whose `author` matches bot patterns (-bot, bot-, _bot, bot_)
        chunk_no_bot_author = chunk_no_bot_body[~chunk_no_bot_body['author'].apply(is_bot_author)]
        bot_author_count = len(chunk_no_bot_body) - len(chunk_no_bot_author)
        bot_author_rows += bot_author_count
        
        # Step 3: keep English rows only
        chunk_english = chunk_no_bot_author[chunk_no_bot_author['body'].apply(is_english)]
        english_count = len(chunk_english)
        english_rows += english_count
        
        # Step 4: keep rows where word_count > 15
        # Ensure word_count is numeric
        chunk_english = chunk_english.copy()
        chunk_english['word_count'] = pd.to_numeric(chunk_english['word_count'], errors='coerce')
        chunk_word_count = chunk_english[chunk_english['word_count'] > 15]
        word_count_filtered_count = len(chunk_english) - len(chunk_word_count)
        word_count_filtered += word_count_filtered_count
        
        # Step 5: date filter (keep on/after 2020-01-20)
        chunk_date_filtered = chunk_word_count[chunk_word_count['created'].apply(lambda x: is_valid_date(x, min_date))]
        date_filtered_count = len(chunk_word_count) - len(chunk_date_filtered)
        date_filtered += date_filtered_count
        
        # Step 6: dedup optimization
        chunk_date_filtered = chunk_date_filtered.copy()  # avoid SettingWithCopyWarning
        
        # Use MD5 for a stable dedup key
        def get_body_hash(text):
            """Generate an MD5 hash for `body` (case-insensitive, trims whitespace)."""
            if pd.isna(text):
                text_str = ''
            else:
                # Normalize: string, trim, lowercase. This treats case-only or
                # leading/trailing-whitespace differences as duplicates.
                text_str = str(text).strip().lower()
            return hashlib.md5(text_str.encode('utf-8')).hexdigest()
        
        # Compute hashes for this chunk
        chunk_date_filtered['body_hash'] = chunk_date_filtered['body'].apply(get_body_hash)
        
        # Deduplicate within the chunk by hash (case-insensitive)
        chunk_date_filtered = chunk_date_filtered.drop_duplicates(subset=['body_hash'], keep='first')
        
        # Keep only rows whose hash has not been seen before
        chunk_deduplicated = chunk_date_filtered[~chunk_date_filtered['body_hash'].isin(seen_body_hashes)]
        
        # Update the seen-hash set (only with kept rows)
        seen_body_hashes.update(chunk_deduplicated['body_hash'].unique())
        
        # Drop the temporary hash column
        chunk_deduplicated = chunk_deduplicated.drop(columns=['body_hash'])
        
        unique_rows += len(chunk_deduplicated)
        
        # Write deduplicated data
        if first_chunk:
            # Include header on the first write
            chunk_deduplicated.to_csv(f_out, index=False, mode='w')
            first_chunk = False
        else:
            # Subsequent writes without header
            chunk_deduplicated.to_csv(f_out, index=False, mode='a', header=False)
        
        # Progress
        print(
            f"Processed {total_rows:,} rows | Removed bot messages: {bot_body_rows:,} | "
            f"Removed bot authors: {bot_author_rows:,} | English: {english_rows:,} | "
            f"word_count filtered: {word_count_filtered:,} | Date filtered: {date_filtered:,} | "
            f"Kept after dedup: {unique_rows:,} | Hashes recorded: {len(seen_body_hashes):,}"
        )

print("\nDone!")
print(f"Total rows: {total_rows:,}")
print(f"Removed bot messages (body): {bot_body_rows:,}")
print(f"Removed bot authors (author): {bot_author_rows:,}")
print(f"English rows: {english_rows:,}")
print(f"word_count filtered (<=15): {word_count_filtered:,}")
print(f"Date filtered (<2020-01-20): {date_filtered:,}")
print(f"Rows kept after dedup: {unique_rows:,}")
print(
    f"Total removed: {total_rows - unique_rows:,} "
    f"(bot message + bot author + non-English + word_count<=15 + date<2020-01-20 + duplicates)"
)
print(f"Output file: {output_file}")

# If `climate` and `covid` exist, compute daily stats
if has_climate_covid:
    print("\nStarting daily stats...")
    stats_output_file = output_file.replace('.csv', '_daily_stats.csv')
    
    # Collect chunks for stats
    all_data_for_stats = []
    
    # Re-read the filtered file for stats
    print("Reading the filtered output for stats...")
    for chunk in pd.read_csv(output_file, chunksize=chunk_size, low_memory=False):
        # Ensure climate/covid are numeric
        chunk['climate'] = pd.to_numeric(chunk['climate'], errors='coerce').fillna(0)
        chunk['covid'] = pd.to_numeric(chunk['covid'], errors='coerce').fillna(0)
        
        # Parse datetime
        chunk['date'] = pd.to_datetime(chunk['created'], errors='coerce')
        
        # Keep rows with valid dates only
        chunk_valid = chunk[chunk['date'].notna()].copy()
        
        # Extract date part (drop time)
        chunk_valid['date_only'] = chunk_valid['date'].dt.date
        
        # Grouping: climate (1,0), covid (0,1), both (1,1)
        chunk_valid['group'] = 'none'
        chunk_valid.loc[(chunk_valid['climate'] == 1) & (chunk_valid['covid'] == 0), 'group'] = 'climate'
        chunk_valid.loc[(chunk_valid['climate'] == 0) & (chunk_valid['covid'] == 1), 'group'] = 'covid'
        chunk_valid.loc[(chunk_valid['climate'] == 1) & (chunk_valid['covid'] == 1), 'group'] = 'both'
        
        # Keep classified rows only
        chunk_valid = chunk_valid[chunk_valid['group'] != 'none']
        
        # Append to list
        all_data_for_stats.append(chunk_valid[['date_only', 'group']])
    
    if all_data_for_stats:
        # Merge all data
        stats_df = pd.concat(all_data_for_stats, ignore_index=True)
        
        # Count by date and group
        daily_stats = stats_df.groupby(['date_only', 'group']).size().reset_index(name='count')
        
        # Pivot: date as rows, group as columns
        pivot_stats = daily_stats.pivot(index='date_only', columns='group', values='count').fillna(0)
        
        # Ensure all groups exist as columns
        for group in ['climate', 'covid', 'both']:
            if group not in pivot_stats.columns:
                pivot_stats[group] = 0
        
        # Reorder columns: climate, covid, both
        pivot_stats = pivot_stats[['climate', 'covid', 'both']]
        
        # Reset index so date becomes a column
        pivot_stats = pivot_stats.reset_index()
        pivot_stats.columns.name = None
        
        # Rename columns
        pivot_stats.columns = ['date', 'climate', 'covid', 'both']
        
        # Sort by date
        pivot_stats = pivot_stats.sort_values('date')
        
        # Convert date to string
        pivot_stats['date'] = pivot_stats['date'].astype(str)
        
        # Convert counts to int
        pivot_stats['climate'] = pivot_stats['climate'].astype(int)
        pivot_stats['covid'] = pivot_stats['covid'].astype(int)
        pivot_stats['both'] = pivot_stats['both'].astype(int)
        
        # Save stats
        pivot_stats.to_csv(stats_output_file, index=False, encoding='utf-8')
        
        print("\nDaily stats complete!")
        print(f"Stats file: {stats_output_file}")
        print(f"Date range: {pivot_stats['date'].min()} to {pivot_stats['date'].max()}")
        print(f"Total dates: {len(pivot_stats)}")
        print("\nFirst 10 rows:")
        print(pivot_stats.head(10).to_string(index=False))
    else:
        print("Warning: no valid data found for stats.")
