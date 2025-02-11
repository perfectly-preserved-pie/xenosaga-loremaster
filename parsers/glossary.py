import re
import json

def parse_glossary(file_path: str) -> list:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Regex to capture:
    # - A header block delimited by a dashed line.
    # - The header line: term and id.
    # - A following dashed line.
    # - Then the definition until the next dashed line or end-of-file.
    pattern = r"(?ms)^-+\s*\n\s*(.+?)\s+(\d+)\s*\n-+\s*\n(.*?)(?=\n-+|\Z)"
    matches = re.findall(pattern, text)

    structured_entries = []
    for term, id_code, definition in matches:
        structured_entries.append({
             "term": term.strip(),
             "id": id_code.strip(),
             "definition": definition.strip(),
             "source": "glossary",
             "source_url": "https://gamefaqs.gamespot.com/ps2/519264-xenosaga-episode-i-der-wille-zur-macht/faqs/47715",
             "episode": "Xenosaga I",
        })
    
    return structured_entries

if __name__ == "__main__":
    entries = parse_glossary("datasets/glossary.txt")
    with open("glossary_entries.json", "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)