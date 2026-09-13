import re
from pathlib import Path


DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 120


def _split_sentences(text):
    """
    Split text into sentences while preserving readable boundaries.
    """
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def _split_words(text, chunk_size):
    """
    Word-level fallback for very large sentences.
    """
    words = text.split()

    chunks = []
    current = []

    for word in words:

        candidate = " ".join(current + [word])

        if len(candidate) <= chunk_size:
            current.append(word)

        else:
            if current:
                chunks.append(" ".join(current))

            current = [word]

    if current:
        chunks.append(" ".join(current))

    return chunks


def _recursive_split(
    text,
    chunk_size=DEFAULT_CHUNK_SIZE
):
    """
    Structure-aware recursive splitting:

    paragraph
        ↓
    sentence
        ↓
    word fallback
    """

    text = text.strip()

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    # --------------------------------------------------------
    # First: paragraphs
    # --------------------------------------------------------

    paragraphs = re.split(
        r"\n\s*\n+",
        text
    )

    paragraphs = [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:

        candidate = (
            paragraph
            if not current
            else current + "\n\n" + paragraph
        )

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        # ----------------------------------------------------
        # Paragraph itself is too large → sentences
        # ----------------------------------------------------

        if len(paragraph) <= chunk_size:
            current = paragraph
            continue

        sentences = _split_sentences(paragraph)

        sentence_buffer = ""

        for sentence in sentences:

            candidate = (
                sentence
                if not sentence_buffer
                else sentence_buffer + " " + sentence
            )

            if len(candidate) <= chunk_size:
                sentence_buffer = candidate

            else:

                if sentence_buffer:
                    chunks.append(sentence_buffer)

                sentence_buffer = ""

                # ------------------------------------------------
                # Sentence itself is too large → words
                # ------------------------------------------------

                if len(sentence) > chunk_size:
                    word_chunks = _split_words(
                        sentence,
                        chunk_size
                    )

                    chunks.extend(word_chunks)

                else:
                    sentence_buffer = sentence

        if sentence_buffer:
            current = sentence_buffer

    if current:
        chunks.append(current)

    return chunks


def _add_overlap(chunks, overlap):
    """
    Add character-based overlap between consecutive chunks.
    """

    if not chunks:
        return []

    if overlap <= 0:
        return chunks

    result = []

    for i, chunk in enumerate(chunks):

        if i == 0:
            result.append(chunk)
            continue

        previous = chunks[i - 1]

        overlap_text = previous[-overlap:]

        combined = (
            overlap_text.rstrip()
            + "\n"
            + chunk.lstrip()
        )

        result.append(combined)

    return result


def chunk_text(
    text,
    chunk_size=DEFAULT_CHUNK_SIZE,
    overlap=DEFAULT_OVERLAP
):
    """
    Chunk a single document into structure-aware chunks.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0"
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative"
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    raw_chunks = _recursive_split(
        text,
        chunk_size=chunk_size
    )

    return _add_overlap(
        raw_chunks,
        overlap=overlap
    )


def chunk_documents(
    documents,
    chunk_size=DEFAULT_CHUNK_SIZE,
    overlap=DEFAULT_OVERLAP
):
    """
    Chunk multiple documents.

    Expected input:

        [
            {
                "text": "...",
                "source_doc": "sample.txt"
            }
        ]

    Returns:

        [
            {
                "chunk_id": "chunk_0001",
                "text": "...",
                "source_doc": "sample.txt",
                "char_start": 0,
                "char_end": 800
            }
        ]
    """

    all_chunks = []

    for document in documents:

        text = document.get("text", "")
        source_doc = document.get(
            "source_doc",
            "unknown"
        )

        if not text:
            continue

        raw_chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        search_position = 0

        for raw_chunk in raw_chunks:

            # Locate the chunk in the original document.
            clean_chunk = raw_chunk.strip()

            char_start = text.find(
                clean_chunk,
                search_position
            )

            if char_start == -1:
                char_start = search_position

            char_end = char_start + len(clean_chunk)

            all_chunks.append(
                {
                    "chunk_id": (
                        f"chunk_{len(all_chunks) + 1:04d}"
                    ),
                    "text": clean_chunk,
                    "source_doc": source_doc,
                    "char_start": char_start,
                    "char_end": char_end,
                }
            )

            search_position = min(
                char_end,
                len(text)
            )

    return all_chunks


def chunk_directory(
    directory,
    chunk_size=DEFAULT_CHUNK_SIZE,
    overlap=DEFAULT_OVERLAP
):
    """
    Convenience function for chunking every .txt file
    inside a directory.
    """

    directory = Path(directory)

    documents = []

    for file_path in sorted(directory.glob("*.txt")):

        text = file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        documents.append(
            {
                "text": text,
                "source_doc": file_path.name,
            }
        )

    return chunk_documents(
        documents,
        chunk_size=chunk_size,
        overlap=overlap
    )