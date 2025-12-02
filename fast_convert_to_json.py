"""
Fast conversion from Python dictionary format to JSON format
Uses string replacement for better performance on large files
"""
import re
import os

def fast_convert_line(line):
    """Convert a single line from Python dict to JSON format using regex"""
    # Remove unicode prefix u' and u" from strings
    line = re.sub(r"\bu'", "'", line)
    line = re.sub(r'\bu"', '"', line)

    # Replace single quotes with double quotes (being careful with apostrophes in strings)
    # This regex handles most cases but may not be perfect for all edge cases
    line = re.sub(r":\s*'([^']*)'", r': "\1"', line)  # After colons
    line = re.sub(r"'([^']+)':", r'"\1":', line)  # Keys
    line = re.sub(r"\[\s*'([^']*)'", r'["\1"', line)  # After [
    line = re.sub(r",\s*'([^']*)'", r', "\1"', line)  # After commas

    # Replace Python booleans and None with JSON equivalents
    line = re.sub(r'\bTrue\b', 'true', line)
    line = re.sub(r'\bFalse\b', 'false', line)
    line = re.sub(r'\bNone\b', 'null', line)

    return line

def convert_file(input_file, output_file):
    """Convert a file from Python dict format to JSON format"""
    print(f"Converting {input_file}...")
    count = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            # Fast string-based conversion
            converted = fast_convert_line(line.strip())
            outfile.write(converted + '\n')
            count += 1

            if count % 10000 == 0:
                print(f"  Processed {count:,} lines...")

    print(f"✓ Converted {count:,} records to {output_file}\n")

if __name__ == '__main__':
    data_dir = 'data'

    files_to_convert = [
        'Australian Users Items.json',
        'Australian User Reviews.json',
        'Steam Games Dataset.json'
    ]

    for filename in files_to_convert:
        input_path = os.path.join(data_dir, filename)
        output_path = os.path.join(data_dir, f'json_{filename}')

        if os.path.exists(input_path):
            convert_file(input_path, output_path)
        else:
            print(f"Warning: {input_path} not found, skipping...")

    print("\n" + "="*80)
    print("Conversion complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Verify the converted files work:")
    print("   head -n 1 data/json_*.json")
    print("\n2. If they look good, backup and replace the originals:")
    print("   cd data")
    print("   mkdir -p backup")
    print("   mv '*.json' backup/  # backup originals")
    print("   mv json_*.json .  # move converted files")
    print("   # Then remove 'json_' prefix from filenames")
