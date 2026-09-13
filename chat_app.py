import os
from pathlib import Path

import streamlit as st

from chunking import chunk_documents
from rag_pipeline import RAGPipeline


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RAG System — Retrieval-Augmented Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# DATA / PIPELINE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"


@st.cache_resource(show_spinner="Initializing retrieval system...")
def load_pipeline():
    documents = []

    if DOCUMENTS_DIR.exists():
        for file_path in sorted(DOCUMENTS_DIR.glob("*.txt")):
            try:
                text = file_path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )

                documents.append(
                    {
                        "text": text,
                        "source_doc": file_path.name,
                    }
                )

            except Exception:
                continue

    if not documents:
        raise RuntimeError(
            "No .txt documents were found in data/documents."
        )

    chunks = chunk_documents(documents)

    return RAGPipeline(chunks), len(documents), len(chunks)


try:
    pipeline, document_count, chunk_count = load_pipeline()
    pipeline_error = None
except Exception as exc:
    pipeline = None
    document_count = 0
    chunk_count = 0
    pipeline_error = str(exc)


# ============================================================
# CUSTOM CSS
# ============================================================

st.html(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600&display=swap');


/* ----------------------------------------------------------
   GLOBAL
---------------------------------------------------------- */

:root {
    --paper: #fcfcfa;
    --ink: #111111;
    --muted: #696969;
    --soft: #969696;
    --line: #deded9;
    --line-dark: #c9c9c2;
    --blue: #315cff;
    --blue-soft: #eef2ff;
    --panel: #f5f5f1;
}

html {
    scroll-behavior: smooth;
}

.stApp {
    background:
        linear-gradient(
            rgba(0, 0, 0, 0.026) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(0, 0, 0, 0.026) 1px,
            transparent 1px
        ),
        var(--paper);

    background-size: 48px 48px;
    color: var(--ink);
    font-family: "Inter", sans-serif;
}


/* Remove Streamlit default padding */

.block-container {
    max-width: 1240px;
    padding-top: 0.7rem !important;
    padding-bottom: 4rem !important;
}


/* Hide default Streamlit chrome */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}


/* ----------------------------------------------------------
   NAVIGATION
---------------------------------------------------------- */

.site-nav {
    position: sticky;
    top: 0;
    z-index: 100;

    display: flex;
    align-items: center;
    justify-content: space-between;

    height: 72px;

    border-bottom: 1px solid var(--line);

    background: rgba(252, 252, 250, 0.94);
    backdrop-filter: blur(12px);
}

.nav-brand {
    display: flex;
    align-items: center;
    gap: 10px;

    color: var(--ink);
    text-decoration: none;
}

.nav-mark {
    width: 26px;
    height: 26px;

    display: flex;
    align-items: center;
    justify-content: center;

    border: 1px solid var(--ink);

    font-family: "JetBrains Mono", monospace;
    font-size: 11px;
    font-weight: 500;
}

.nav-title {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.nav-links {
    display: flex;
    align-items: center;
    gap: 32px;
}

.nav-links a {
    color: #555;
    text-decoration: none;

    font-family: "JetBrains Mono", monospace;
    font-size: 11px;
    letter-spacing: 0.04em;
    text-transform: uppercase;

    transition:
        color 0.2s ease,
        transform 0.2s ease;
}

.nav-links a:hover {
    color: var(--ink);
    transform: translateY(-1px);
}


/* ----------------------------------------------------------
   HERO
---------------------------------------------------------- */

.hero {
    min-height: 650px;

    display: grid;
    grid-template-columns: 1.05fr 0.95fr;
    gap: 70px;

    align-items: center;

    border-bottom: 1px solid var(--line);
}

.eyebrow {
    margin-bottom: 22px;

    color: var(--blue);

    font-family: "JetBrains Mono", monospace;
    font-size: 11px;
    font-weight: 500;

    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.hero-title {
    max-width: 690px;

    margin: 0;

    font-family: "Source Serif 4", serif;
    font-size: clamp(58px, 7vw, 94px);
    font-weight: 500;

    line-height: 0.94;
    letter-spacing: -0.055em;
}

.hero-title em {
    color: var(--blue);
    font-style: normal;
}

.hero-description {
    max-width: 570px;

    margin-top: 32px;

    color: var(--muted);

    font-size: 16px;
    line-height: 1.75;
}

.hero-meta {
    display: flex;
    gap: 35px;

    margin-top: 42px;
}

.meta-item {
    display: flex;
    flex-direction: column;
    gap: 6px;
}

.meta-label {
    color: var(--soft);

    font-family: "JetBrains Mono", monospace;
    font-size: 9px;

    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.meta-value {
    font-family: "JetBrains Mono", monospace;
    font-size: 12px;
    font-weight: 500;
}

.hero-actions {
    display: flex;
    gap: 12px;

    margin-top: 38px;
}

.hero-action {
    display: inline-flex;
    align-items: center;
    justify-content: center;

    min-width: 132px;
    height: 44px;

    border: 1px solid var(--ink);

    color: var(--ink);
    background: transparent;

    text-decoration: none;

    font-family: "JetBrains Mono", monospace;
    font-size: 10px;

    letter-spacing: 0.06em;
    text-transform: uppercase;

    transition: all 0.2s ease;
}

.hero-action.primary {
    color: white;
    background: var(--ink);
}

.hero-action:hover {
    transform: translateY(-2px);
}

.hero-action.primary:hover {
    background: #252525;
}


/* ----------------------------------------------------------
   RETRIEVAL DIAGRAM
---------------------------------------------------------- */

.diagram-wrap {
    position: relative;

    min-height: 430px;

    display: flex;
    align-items: center;
    justify-content: center;
}

.diagram-frame {
    width: 100%;
    max-width: 510px;

    padding: 26px;

    border: 1px solid var(--line-dark);

    background: rgba(255, 255, 255, 0.55);
}

.diagram-top {
    display: flex;
    align-items: center;
    justify-content: space-between;

    margin-bottom: 28px;

    padding-bottom: 14px;

    border-bottom: 1px solid var(--line);
}

.diagram-label {
    font-family: "JetBrains Mono", monospace;
    font-size: 9px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.diagram-status {
    display: flex;
    align-items: center;
    gap: 7px;

    font-family: "JetBrains Mono", monospace;
    font-size: 9px;
    color: var(--muted);
}

.status-dot {
    width: 6px;
    height: 6px;

    border-radius: 50%;
    background: var(--blue);
}

.rag-diagram {
    width: 100%;
}

.diagram-node {
    fill: var(--paper);
    stroke: #222;
    stroke-width: 1.1;
}

.diagram-node-accent {
    fill: var(--blue-soft);
    stroke: var(--blue);
    stroke-width: 1.1;
}

.diagram-text {
    fill: #222;
    font-family: "JetBrains Mono", monospace;
    font-size: 10px;
}

.diagram-small {
    fill: #777;
    font-family: "JetBrains Mono", monospace;
    font-size: 7px;
}

.diagram-line {
    fill: none;
    stroke: #aaa;
    stroke-width: 1;
}

.diagram-line-accent {
    fill: none;
    stroke: var(--blue);
    stroke-width: 1.3;
}


/* ----------------------------------------------------------
   SECTION STRUCTURE
---------------------------------------------------------- */

.section {
    padding: 105px 0;

    border-bottom: 1px solid var(--line);
}

.section-header {
    display: grid;
    grid-template-columns: 0.35fr 1fr;
    gap: 55px;

    margin-bottom: 70px;
}

.section-index {
    color: var(--blue);

    font-family: "JetBrains Mono", monospace;
    font-size: 11px;
    letter-spacing: 0.1em;
}

.section-title {
    margin: 0;

    font-family: "Source Serif 4", serif;
    font-size: clamp(42px, 5vw, 68px);
    font-weight: 500;

    line-height: 0.98;
    letter-spacing: -0.045em;
}

.section-subtitle {
    max-width: 650px;

    margin-top: 20px;

    color: var(--muted);

    font-size: 15px;
    line-height: 1.7;
}


/* ----------------------------------------------------------
   PIPELINE
---------------------------------------------------------- */

.pipeline-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
}

.pipeline-step {
    position: relative;

    min-height: 190px;

    padding: 20px;

    border-top: 1px solid var(--line);
    border-right: 1px solid var(--line);
}

.pipeline-step:first-child {
    border-left: 1px solid var(--line);
}

.pipeline-step::after {
    content: "";

    position: absolute;

    top: 54px;
    right: -1px;

    width: 20px;
    height: 1px;

    background: var(--line-dark);
}

.pipeline-step:last-child::after {
    display: none;
}

.step-number {
    color: var(--blue);

    font-family: "JetBrains Mono", monospace;
    font-size: 10px;
}

.step-icon {
    width: 34px;
    height: 34px;

    display: flex;
    align-items: center;
    justify-content: center;

    margin-top: 20px;
    margin-bottom: 18px;

    border: 1px solid var(--line-dark);

    font-family: "JetBrains Mono", monospace;
    font-size: 11px;
}

.step-title {
    margin-bottom: 7px;

    font-size: 13px;
    font-weight: 600;

    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.step-description {
    color: var(--muted);

    font-family: "JetBrains Mono", monospace;
    font-size: 9px;
    line-height: 1.6;
}


/* ----------------------------------------------------------
   ENGINEERING / FEATURES
---------------------------------------------------------- */

.engineering-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
}

.feature {
    min-height: 235px;

    padding: 28px;

    border-top: 1px solid var(--line);
    border-right: 1px solid var(--line);
}

.feature:nth-child(3n + 1) {
    border-left: 1px solid var(--line);
}

.feature-number {
    color: var(--blue);

    font-family: "JetBrains Mono", monospace;
    font-size: 10px;
}

.feature h3 {
    margin: 30px 0 13px;

    font-family: "Source Serif 4", serif;
    font-size: 28px;
    font-weight: 500;

    letter-spacing: -0.025em;
}

.feature p {
    margin: 0;

    color: var(--muted);

    font-size: 13px;
    line-height: 1.7;
}


/* ----------------------------------------------------------
   TRY IT
---------------------------------------------------------- */

.try-section {
    padding: 105px 0 80px;
}

.query-panel {
    border: 1px solid var(--line-dark);

    background: rgba(255, 255, 255, 0.62);
}

.query-panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;

    padding: 20px 24px;

    border-bottom: 1px solid var(--line);
}

.query-panel-title {
    font-family: "JetBrains Mono", monospace;
    font-size: 10px;

    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.query-panel-status {
    color: var(--muted);

    font-family: "JetBrains Mono", monospace;
    font-size: 9px;
}

.query-panel-body {
    padding: 28px;
}


/* ----------------------------------------------------------
   TRY QUESTIONS
---------------------------------------------------------- */

.try-questions {
    display: none;

    align-items: center;
    gap: 8px;
    flex-wrap: wrap;

    margin: 0 0 10px 0;
    padding: 0 2px;

    animation: tryQuestionsIn 0.18s ease-out;
}

.try-label {
    color: var(--soft);

    font-family: "JetBrains Mono", monospace;
    font-size: 9px;

    letter-spacing: 0.08em;
    text-transform: uppercase;

    margin-right: 2px;
}

.try-question {
    display: inline-flex;
    align-items: center;

    min-height: 28px;

    padding: 5px 10px;

    border: 1px solid var(--line-dark);

    color: #555;
    background: rgba(255, 255, 255, 0.7);

    font-family: "JetBrains Mono", monospace;
    font-size: 9px;

    line-height: 1.2;

    transition:
        border-color 0.18s ease,
        color 0.18s ease,
        transform 0.18s ease;
}

.try-question:hover {
    color: var(--ink);
    border-color: #999;

    transform: translateY(-1px);
}

@keyframes tryQuestionsIn {
    from {
        opacity: 0;
        transform: translateY(3px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}


/*
Reveal suggestions when the Streamlit chat input
receives focus / typing.
*/

body:has(
    div[data-testid="stChatInput"]:focus-within
) .try-questions {
    display: flex;
}


/* ----------------------------------------------------------
   STREAMLIT CHAT INPUT
---------------------------------------------------------- */

div[data-testid="stChatInput"] {
    margin-top: 0 !important;
}

div[data-testid="stChatInput"] > div {
    border: 1px solid var(--line-dark) !important;

    border-radius: 0 !important;

    background: white !important;

    box-shadow: none !important;
}

div[data-testid="stChatInput"] textarea {
    font-family: "Inter", sans-serif !important;
    font-size: 14px !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
    color: #999 !important;
}


/* ----------------------------------------------------------
   CHAT MESSAGES
---------------------------------------------------------- */

div[data-testid="stChatMessage"] {
    border-bottom: 1px solid var(--line);

    border-radius: 0 !important;

    padding-top: 22px;
    padding-bottom: 22px;
}

div[data-testid="stChatMessage"] p {
    font-size: 14px;
    line-height: 1.75;
}

div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    color: #242424;
}


/* ----------------------------------------------------------
   SOURCE EXPANDERS
---------------------------------------------------------- */

div[data-testid="stExpander"] {
    border: 1px solid var(--line) !important;

    border-radius: 0 !important;

    background: rgba(255, 255, 255, 0.55);
}

div[data-testid="stExpander"] summary {
    font-family: "JetBrains Mono", monospace !important;
    font-size: 10px !important;
}


/* ----------------------------------------------------------
   SELECTBOX
---------------------------------------------------------- */

div[data-baseweb="select"] > div {
    border-radius: 0 !important;

    border-color: var(--line-dark) !important;

    background: white !important;

    box-shadow: none !important;
}


/* ----------------------------------------------------------
   FOOTER
---------------------------------------------------------- */

.site-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;

    padding: 30px 0;

    color: var(--soft);

    font-family: "JetBrains Mono", monospace;
    font-size: 9px;

    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.footer-right {
    color: #aaa;
}


/* ----------------------------------------------------------
   RESPONSIVE
---------------------------------------------------------- */

@media (max-width: 900px) {

    .hero {
        grid-template-columns: 1fr;
        gap: 45px;

        padding: 70px 0;
    }

    .diagram-wrap {
        min-height: auto;
    }

    .section-header {
        grid-template-columns: 1fr;
        gap: 20px;
    }

    .pipeline-grid {
        grid-template-columns: repeat(2, 1fr);
    }

    .pipeline-step {
        border-left: 1px solid var(--line);
    }

    .pipeline-step::after {
        display: none;
    }

    .engineering-grid {
        grid-template-columns: 1fr;
    }

    .feature,
    .feature:nth-child(3n + 1) {
        border-left: 1px solid var(--line);
    }

    .nav-links {
        gap: 15px;
    }
}


@media (max-width: 600px) {

    .block-container {
        padding-left: 20px !important;
        padding-right: 20px !important;
    }

    .site-nav {
        height: 62px;
    }

    .nav-links a:nth-child(2) {
        display: none;
    }

    .hero {
        min-height: auto;
        padding: 65px 0;
    }

    .hero-title {
        font-size: 58px;
    }

    .hero-meta {
        gap: 20px;
        flex-wrap: wrap;
    }

    .hero-actions {
        flex-wrap: wrap;
    }

    .pipeline-grid {
        grid-template-columns: 1fr;
    }

    .section,
    .try-section {
        padding: 75px 0;
    }

    .query-panel-body {
        padding: 18px;
    }

    .site-footer {
        flex-direction: column;
        align-items: flex-start;
        gap: 12px;
    }
}

</style>
"""
)


# ============================================================
# NAVIGATION
# ============================================================

st.html(
    """
<div class="site-nav">

    <a class="nav-brand" href="#top">
        <div class="nav-mark">R</div>
        <div class="nav-title">RAG System</div>
    </a>

    <div class="nav-links">
        <a href="#pipeline">Pipeline</a>
        <a href="#engineering">Engineering</a>
        <a href="#try-it">Try It</a>
    </div>

</div>

<div id="top"></div>
"""
)


# ============================================================
# HERO
# ============================================================

st.html(
    f"""
<section class="hero">

    <div>

        <div class="eyebrow">
            Retrieval-Augmented Intelligence / 01
        </div>

        <h1 class="hero-title">
            Ask the<br>
            <em>documents.</em>
        </h1>

        <p class="hero-description">
            A modular retrieval-augmented generation system that combines
            dense semantic search, lexical retrieval, reciprocal rank fusion,
            and cross-encoder reranking to produce grounded answers with
            source-level citations.
        </p>

        <div class="hero-meta">

            <div class="meta-item">
                <span class="meta-label">Documents</span>
                <span class="meta-value">{document_count:02d}</span>
            </div>

            <div class="meta-item">
                <span class="meta-label">Indexed Chunks</span>
                <span class="meta-value">{chunk_count:03d}</span>
            </div>

            <div class="meta-item">
                <span class="meta-label">Retrieval</span>
                <span class="meta-value">Hybrid</span>
            </div>

        </div>

        <div class="hero-actions">

            <a class="hero-action primary" href="#try-it">
                Try the system
            </a>

            <a class="hero-action" href="#pipeline">
                View pipeline
            </a>

        </div>

    </div>


    <div class="diagram-wrap">

        <div class="diagram-frame">

            <div class="diagram-top">

                <span class="diagram-label">
                    Retrieval Architecture
                </span>

                <span class="diagram-status">
                    <span class="status-dot"></span>
                    SYSTEM READY
                </span>

            </div>


            <svg
                class="rag-diagram"
                viewBox="0 0 500 350"
                xmlns="http://www.w3.org/2000/svg"
            >

                <!-- Input -->

                <rect
                    x="180"
                    y="15"
                    width="140"
                    height="45"
                    rx="0"
                    class="diagram-node"
                />

                <text
                    x="250"
                    y="34"
                    text-anchor="middle"
                    class="diagram-small"
                >
                    USER QUERY
                </text>

                <text
                    x="250"
                    y="49"
                    text-anchor="middle"
                    class="diagram-text"
                >
                    "query"
                </text>


                <!-- Lines -->

                <path
                    d="M250 60 L250 85"
                    class="diagram-line-accent"
                />


                <!-- Dense -->

                <rect
                    x="45"
                    y="85"
                    width="165"
                    height="70"
                    rx="0"
                    class="diagram-node-accent"
                />

                <text
                    x="127"
                    y="108"
                    text-anchor="middle"
                    class="diagram-small"
                >
                    SEMANTIC
                </text>

                <text
                    x="127"
                    y="127"
                    text-anchor="middle"
                    class="diagram-text"
                >
                    FAISS / Dense
                </text>

                <text
                    x="127"
                    y="143"
                    text-anchor="middle"
                    class="diagram-small"
                >
                    top-k candidates
                </text>


                <!-- BM25 -->

                <rect
                    x="290"
                    y="85"
                    width="165"
                    height="70"
                    rx="0"
                    class="diagram-node"
                />

                <text
                    x="372"
                    y="108"
                    text-anchor="middle"
                    class="diagram-small"
                >
                    LEXICAL
                </text>

                <text
                    x="372"
                    y="127"
                    text-anchor="middle"
                    class="diagram-text"
                >
                    BM25
                </text>

                <text
                    x="372"
                    y="143"
                    text-anchor="middle"
                    class="diagram-small"
                >
                    top-k candidates
                </text>


                <!-- Merge lines -->

                <path
                    d="M127 155 L127 190 L250 190"
                    class="diagram-line"
                />

                <path
                    d="M372 155 L372 190 L250 190"
                    class="diagram-line"
                />


                <!-- RRF -->

                <rect
                    x="165"
                    y="190"
                    width="170"
                    height="60"
                    rx="0"
                    class="diagram-node"
                />

                <text
                    x="250"
                    y="213"
                    text-anchor="middle"
                    class="diagram-small"
                >
                    FUSION
                </text>

                <text
                    x="250"
                    y="232"
                    text-anchor="middle"
                    class="diagram-text"
                >
                    Reciprocal Rank Fusion
                </text>


                <!-- Rerank -->

                <path
                    d="M250 250 L250 275"
                    class="diagram-line-accent"
                />

                <rect
                    x="165"
                    y="275"
                    width="170"
                    height="55"
                    rx="0"
                    class="diagram-node-accent"
                />

                <text
                    x="250"
                    y="296"
                    text-anchor="middle"
                    class="diagram-small"
                >
                    RERANK
                </text>

                <text
                    x="250"
                    y="315"
                    text-anchor="middle"
                    class="diagram-text"
                >
                    Cross-Encoder → Top 3
                </text>

            </svg>

        </div>

    </div>

</section>
"""
)


# ============================================================
# PIPELINE SECTION
# ============================================================

st.html(
    """
<section class="section" id="pipeline">

    <div class="section-header">

        <div class="section-index">
            02 / PIPELINE
        </div>

        <div>

            <h2 class="section-title">
                From raw text<br>
                to grounded answer.
            </h2>

            <p class="section-subtitle">
                Every stage is isolated as a modular component so retrieval,
                fusion, reranking, and generation can be inspected,
                evaluated, and replaced independently.
            </p>

        </div>

    </div>


    <div class="pipeline-grid">

        <div class="pipeline-step">

            <div class="step-number">01</div>

            <div class="step-icon">↓</div>

            <div class="step-title">Ingest</div>

            <div class="step-description">
                Load source documents
                from the knowledge base.
            </div>

        </div>


        <div class="pipeline-step">

            <div class="step-number">02</div>

            <div class="step-icon">#</div>

            <div class="step-title">Chunk</div>

            <div class="step-description">
                Structure-aware splitting
                with overlap and metadata.
            </div>

        </div>


        <div class="pipeline-step">

            <div class="step-number">03</div>

            <div class="step-icon">⌕</div>

            <div class="step-title">Retrieve</div>

            <div class="step-description">
                FAISS semantic retrieval
                plus BM25 lexical search.
            </div>

        </div>


        <div class="pipeline-step">

            <div class="step-number">04</div>

            <div class="step-icon">⊕</div>

            <div class="step-title">Fuse</div>

            <div class="step-description">
                Reciprocal Rank Fusion
                combines candidate rankings.
            </div>

        </div>


        <div class="pipeline-step">

            <div class="step-number">05</div>

            <div class="step-icon">↕</div>

            <div class="step-title">Rerank</div>

            <div class="step-description">
                Cross-encoder scoring selects
                the most relevant context.
            </div>

        </div>


        <div class="pipeline-step">

            <div class="step-number">06</div>

            <div class="step-icon">→</div>

            <div class="step-title">Generate</div>

            <div class="step-description">
                Grounded response generation
                with chunk-level citations.
            </div>

        </div>

    </div>

</section>
"""
)


# ============================================================
# ENGINEERING SECTION
# ============================================================

st.html(
    """
<section class="section" id="engineering">

    <div class="section-header">

        <div class="section-index">
            03 / ENGINEERING
        </div>

        <div>

            <h2 class="section-title">
                Built for<br>
                inspectability.
            </h2>

            <p class="section-subtitle">
                The system separates each retrieval concern into a dedicated
                module rather than hiding the entire RAG workflow behind
                a single framework abstraction.
            </p>

        </div>

    </div>


    <div class="engineering-grid">

        <article class="feature">

            <div class="feature-number">01</div>

            <h3>Intelligent Chunking</h3>

            <p>
                Recursive, structure-aware document splitting with
                configurable overlap and source metadata tracking.
            </p>

        </article>


        <article class="feature">

            <div class="feature-number">02</div>

            <h3>Hybrid Retrieval</h3>

            <p>
                Dense semantic retrieval with FAISS is combined with
                lexical BM25 search for broader query coverage.
            </p>

        </article>


        <article class="feature">

            <div class="feature-number">03</div>

            <h3>Cross-Encoder Reranking</h3>

            <p>
                Hybrid candidates are reranked with a cross-encoder
                before the final context is passed to generation.
            </p>

        </article>


        <article class="feature">

            <div class="feature-number">04</div>

            <h3>Grounded Generation</h3>

            <p>
                The generation layer is instructed to rely only on the
                retrieved context and expose supporting chunk citations.
            </p>

        </article>


        <article class="feature">

            <div class="feature-number">05</div>

            <h3>Modular Architecture</h3>

            <p>
                Chunking, retrieval, fusion, reranking, and generation
                are isolated into independently understandable modules.
            </p>

        </article>


        <article class="feature">

            <div class="feature-number">06</div>

            <h3>Source Transparency</h3>

            <p>
                Retrieved chunks remain visible beneath answers so the
                relationship between evidence and response can be inspected.
            </p>

        </article>

    </div>

</section>
"""
)


# ============================================================
# TRY IT SECTION
# ============================================================

st.html(
    """
<section class="try-section" id="try-it">

    <div class="section-header">

        <div class="section-index">
            04 / INTERACTION
        </div>

        <div>

            <h2 class="section-title">
                Ask the<br>
                knowledge base.
            </h2>

            <p class="section-subtitle">
                Choose a retrieval strategy, ask a question, and inspect
                the evidence used to construct the answer.
            </p>

        </div>

    </div>


    <div class="query-panel">

        <div class="query-panel-header">

            <span class="query-panel-title">
                Interactive RAG Console
            </span>

            <span class="query-panel-status">
                LOCAL DOCUMENT INDEX
            </span>

        </div>

        <div class="query-panel-body">
"""
)


# ============================================================
# PIPELINE ERROR
# ============================================================

if pipeline_error:

    st.error(
        f"RAG pipeline could not be initialized: {pipeline_error}"
    )

else:

    # --------------------------------------------------------
    # RETRIEVAL STRATEGY
    # --------------------------------------------------------

    strategy_labels = {
        "hybrid_reranked": "Hybrid + Cross-Encoder Reranking",
        "hybrid_rrf": "Hybrid + Reciprocal Rank Fusion",
        "dense": "Dense / FAISS",
        "bm25": "BM25 / Lexical",
    }

    strategy = st.selectbox(
        "Retrieval strategy",
        options=list(strategy_labels.keys()),
        format_func=lambda x: strategy_labels[x],
        index=0,
        label_visibility="collapsed",
    )


    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    if "messages" not in st.session_state:
        st.session_state.messages = []


    # --------------------------------------------------------
    # DISPLAY PREVIOUS MESSAGES
    # --------------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if (
                message["role"] == "assistant"
                and message.get("used_chunks")
            ):

                with st.expander(
                    f"Sources used · {len(message['used_chunks'])}"
                ):

                    for chunk in message["used_chunks"]:

                        st.markdown(
                            f"**{chunk['chunk_id']}**  \n"
                            f"`{chunk['source_doc']}`"
                        )

                        st.caption(chunk["text"])


    # --------------------------------------------------------
    # TRY THESE QUESTIONS
    # --------------------------------------------------------

    st.html(
        """
        <div class="try-questions">

            <span class="try-label">
                Try asking:
            </span>

            <span class="try-question">
                Summarize the document
            </span>

            <span class="try-question">
                What are the key concepts?
            </span>

            <span class="try-question">
                Explain a specific section
            </span>

            <span class="try-question">
                What retrieval methods are used?
            </span>

        </div>
        """
    )


    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    query = st.chat_input(
        "Ask a question about the indexed documents..."
    )


    # --------------------------------------------------------
    # PROCESS QUERY
    # --------------------------------------------------------

    if query:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query,
            }
        )

        with st.chat_message("user"):
            st.markdown(query)


        with st.chat_message("assistant"):

            with st.spinner("Retrieving context..."):

                try:

                    result = pipeline.answer(
                        query,
                        strategy=strategy,
                    )

                    answer = result["answer"]
                    used_chunks = result.get(
                        "used_chunks",
                        [],
                    )

                    st.markdown(answer)

                    if used_chunks:

                        with st.expander(
                            f"Sources used · {len(used_chunks)}"
                        ):

                            for chunk in used_chunks:

                                st.markdown(
                                    f"**{chunk['chunk_id']}**  \n"
                                    f"`{chunk['source_doc']}`"
                                )

                                st.caption(
                                    chunk["text"]
                                )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "used_chunks": used_chunks,
                        }
                    )

                except Exception as exc:

                    error_message = (
                        "The query could not be processed. "
                        f"Error: {exc}"
                    )

                    st.error(error_message)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                            "used_chunks": [],
                        }
                    )


# ============================================================
# CLOSE QUERY PANEL / TRY SECTION
# ============================================================

st.html(
    """
        </div>
    </div>

</section>
"""
)


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
<footer class="site-footer">

    <span>
        RAG SYSTEM / MODULAR RETRIEVAL PIPELINE
    </span>

    <span class="footer-right">
        FAISS · BM25 · RRF · CROSS-ENCODER · OPENAI
    </span>

</footer>
"""
)