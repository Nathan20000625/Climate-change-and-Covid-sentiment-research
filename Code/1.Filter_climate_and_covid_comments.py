import pandas as pd
import re
import os
from tqdm import tqdm
import chardet  # add back chardet

# Input and output folder paths
input_folder = "E:/reddit_covid_climate/csv"
output_folder = "E:/reddit_covid_climate/csv"  # output to the same directory

# Regex patterns for climate change and COVID-19
climate_pattern = re.compile(r"(?i)(global\s+warm|global\s+warming|climate\s+change|climate\s+changing|climate\s+crisis|climate\s+shift|climate\s+variation|climate\s+fluctuations|climate\s+emergency|climate\s+instability|atmospheric\s+change|environmental\s+climate\s+transformation|climate\s+related|climate\s+induced|climate\s+affected|climate\s+based|climate\s+driven|disturbing\s+the\s+climate|upsetting\s+the\s+climate\s+balance|modifying\s+the\s+climate|varying\s+the\s+climate|global\s+climate\s+disruption)")
covid_pattern = re.compile(r"(?i)COVID-19|COVID|SARS-CoV-2|coronavirus|pandemic")

# Process CSV files only
for filename in os.listdir(input_folder):
    if filename.endswith(".csv") and not filename.endswith("_filtered.csv"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".csv", "_filtered.csv"))
        chunk_size = 100000  # tune based on available memory

        # Auto-detect encoding
        with open(input_path, 'rb') as f:
            rawdata = f.read(100000)  # read only the first 100k bytes
            result = chardet.detect(rawdata)
            encoding = result['encoding']
        print(f"{filename} detected encoding: {encoding}")

        # Count total rows (excluding header)
        with open(input_path, 'r', encoding=encoding) as f:
            total_lines = sum(1 for _ in f) - 1

        processed_lines = 0
        first_chunk = True
        with tqdm(total=total_lines, desc=f"Processing {filename}", unit="rows") as pbar:
            for chunk in pd.read_csv(input_path, chunksize=chunk_size, dtype=str, encoding=encoding):
                # Ensure `body` column exists
                if 'body' not in chunk.columns:
                    processed_lines += len(chunk)
                    pbar.update(len(chunk))
                    continue

                # Add `climate` and `covid` indicator columns
                chunk['climate'] = chunk['body'].apply(lambda x: 1 if pd.notnull(x) and climate_pattern.search(x) else 0)
                chunk['covid'] = chunk['body'].apply(lambda x: 1 if pd.notnull(x) and covid_pattern.search(x) else 0)

                # Keep rows where climate==1 or covid==1
                filtered_chunk = chunk[(chunk['climate'] == 1) | (chunk['covid'] == 1)]

                # Append-write to output file
                if not filtered_chunk.empty:
                    filtered_chunk.to_csv(output_path, mode='a', index=False, header=first_chunk, encoding='utf-8')
                    first_chunk = False

                processed_lines += len(chunk)
                pbar.update(len(chunk))

        # Summarize output rows and category distribution (excluding header)
        if os.path.exists(output_path):
            # Summarize in chunks to avoid loading large files into memory
            only_climate = 0
            only_covid = 0
            both = 0
            total = 0
            for out_chunk in pd.read_csv(output_path, chunksize=chunk_size, dtype={'climate': int, 'covid': int}, encoding='utf-8'):
                only_climate += ((out_chunk['climate'] == 1) & (out_chunk['covid'] == 0)).sum()
                only_covid += ((out_chunk['climate'] == 0) & (out_chunk['covid'] == 1)).sum()
                both += ((out_chunk['climate'] == 1) & (out_chunk['covid'] == 1)).sum()
                total += len(out_chunk)
            print(f"{filename} finished. Output saved to {output_path}. Total rows: {total}")
            print(f"  climate=1 only: {only_climate} | covid=1 only: {only_covid} | both=1: {both}")
        else:
            print(f"{filename} finished. Output saved to {output_path}. No matching rows.")

print("All files processed!")