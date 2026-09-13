from pathlib import Path
import re


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def split_words(text):
    return text.split()


def merge_units(text, units, chunk_size, overlap):
    chunks = []
    current_text = ""
    current_start = None

    for unit_text, unit_start, unit_end in units:
        candidate = unit_text if not current_text else current_text + "\n\n" + unit_text

        if len(candidate) <= chunk_size:
            if current_start is None:
                current_start = unit_start
            current_text = candidate
        else:
            if current_text:
                chunks.append((current_text, current_start,
                               min(len(text), current_start + len(current_text))))

            overlap_text = current_text[-overlap:] if overlap > 0 else ""
            current_text = (overlap_text + "\n\n" + unit_text
                            if overlap_text else unit_text)
            current_start = max(0, unit_start - len(overlap_text))

    if current_text:
        chunks.append((current_text, current_start,
                       min(len(text), current_start + len(current_text))))

    return chunks


def recursive_split(text, chunk_size, overlap):
    if len(text) <= chunk_size:
        return [(text, 0, len(text))]

    paragraphs = re.split(r"\n\s*\n", text)

    if len(paragraphs) > 1:
        units = []
        cursor = 0
        for paragraph in paragraphs:
            start = text.find(paragraph, cursor)
            end = start + len(paragraph)
            units.append((paragraph, start, end))
            cursor = end
        return merge_units(text, units, chunk_size, overlap)

    sentences = split_sentences(text)

    if len(sentences) > 1:
        units = []
        cursor = 0
        for sentence in sentences:
            start = text.find(sentence, cursor)
            end = start + len(sentence)
            units.append((sentence, start, end))
            cursor = end
        return merge_units(text, units, chunk_size, overlap)

    words = split_words(text)
    chunks = []
    current_words = []
    current_start = 0
    cursor = 0

    for word in words:
        word_start = text.find(word, cursor)
        word_end = word_start + len(word)
        candidate = " ".join(current_words + [word])

        if len(candidate) <= chunk_size:
            current_words.append(word)
        else:
            if current_words:
                chunks.append((" ".join(current_words), current_start, word_start))

            overlap_words = current_words[-max(1, overlap // 10):] if overlap > 0 else []
            overlap_text = " ".join(overlap_words)
            current_words = overlap_words.copy()
            current_start = max(0, word_start - len(overlap_text))
            current_words.append(word)

        cursor = word_end

    if current_words:
        chunks.append((" ".join(current_words), current_start, len(text)))

    return chunks


def chunk_document(text, source_doc, chunk_size=1000, overlap=150):
    return [
        {
            "chunk_id": f"{Path(source_doc).stem}_{i}",
            "text": chunk_text,
            "source_doc": source_doc,
            "char_start": start,
            "char_end": end,
        }
        for i, (chunk_text, start, end)
        in enumerate(recursive_split(text, chunk_size, overlap))
    ]


def chunk_folder(folder_path, chunk_size=1000, overlap=150):
    chunks = []

    for path in sorted(Path(folder_path).glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        chunks.extend(chunk_document(text, path.name, chunk_size, overlap))

    return chunks


if __name__ == "__main__":
    chunks = chunk_folder("data/documents")
    print(f"Created {len(chunks)} chunks.")
    for chunk in chunks[:3]:
        print(chunk)
