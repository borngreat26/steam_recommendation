"""
Convert Python dictionary format files to proper JSON format
This is a one-time conversion script
"""
import ast
import json
import os

def convert_file(input_file, output_file):
    """Convert a file from Python dict format to JSON format"""
    print(f"Converting {input_file}...")
    count = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            # Parse Python dict and convert to JSON
            data = ast.literal_eval(line.strip())
            json.dump(data, outfile)
            outfile.write('\n')
            count += 1

            if count % 1000 == 0:
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
        output_path = os.path.join(data_dir, f'converted_{filename}')

        if os.path.exists(input_path):
            convert_file(input_path, output_path)
        else:
            print(f"Warning: {input_path} not found, skipping...")

    print("Conversion complete!")
    print("\nNext steps:")
    print("1. Check the converted_*.json files")
    print("2. If they look good, rename them to replace the originals:")
    print("   cd data")
    print("   mv 'Australian Users Items.json' 'Australian Users Items.json.bak'")
    print("   mv 'converted_Australian Users Items.json' 'Australian Users Items.json'")
    print("   (repeat for other files)")
