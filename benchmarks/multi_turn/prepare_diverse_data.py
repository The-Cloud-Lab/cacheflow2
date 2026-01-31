import json
from sklearn.datasets import fetch_20newsgroups

def generate_diverse_dataset(output_text_file="diverse_news.txt", output_config="dpu_config_diverse.json"):
    print("Downloading 20 Newsgroups...")
    # Subset='all' gives ~19,000 unique documents
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    
    # Filter out empty or very short posts and combine into one file
    # We take enough to reach ~2.5MB to match pg1184.txt size
    combined_text = []
    current_size = 0
    target_size = 2.5 * 1024 * 1024 # 2.5 MB
    
    for text in newsgroups.data:
        cleaned = text.strip()
        if len(cleaned) > 200:
            combined_text.append(cleaned)
            current_size += len(cleaned)
        if current_size >= target_size:
            break

    # Save the text file
    with open(output_text_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(combined_text))
    
    # Create the new config mirroring your original structure
    config = {
        "filetype": "generate_conversations",
        "num_conversations": 128,
        "text_files": [output_text_file],
        "print_stats": False,
        "prompt_input": {
            "num_turns": {"distribution": "constant", "value": 2},
            "common_prefix_num_tokens": {"distribution": "constant", "value": 0}, # Set to 0 to ensure diversity
            "prefix_num_tokens": {"distribution": "constant", "value": 0},
            "num_tokens": {"distribution": "constant", "value": 512} # Larger tokens to stress the DMA
        },
        "prompt_output": {
            "num_tokens": {"distribution": "uniform", "min": 64, "max": 96}
        }
    }

    with open(output_config, "w") as f:
        json.dump(config, f, indent=4)
    
    print(f"Created {output_text_file} ({current_size/1024/1024:.2f} MB)")
    print(f"Created {output_config}")

if __name__ == "__main__":
    generate_diverse_dataset()