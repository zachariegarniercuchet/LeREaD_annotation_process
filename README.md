# Legal Authority Extraction and Resolution Annotation Framework

[![Research](https://img.shields.io/badge/Status-Research-orange)](https://mila.quebec/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Research-lightgrey)]()

## Overview

This repository contains the annotation framework and prompt engineering pipeline for extracting and resolving legal authorities (references) in Canadian legal decisions. The project is part of ongoing master's research at [Mila - Quebec AI Institute](https://mila.quebec/) investigating the intersection of natural language processing and legal information retrieval.

Legal authorities—citations to prior cases, legislation, and secondary sources—are fundamental to legal reasoning and precedent-based decision-making. Despite their importance, no existing datasets comprehensively address both **extraction** (identifying references in text) and **resolution** (disambiguating co-references and linking to unique legal documents) in the legal NLP literature.

This project aims to:
1. Create a high-quality annotated dataset for legal authority extraction and resolution
2. Evaluate LLM-based annotation approaches using state-of-the-art language models (GPT-4)
3. Compare human annotation, LLM annotation, and collaborative human-LLM workflows
4. Establish benchmarks for future research in legal reference processing

---

## Research Context

### Problem Statement

Legal decisions contain complex citation patterns including:
- **Full citations**: Complete references with case names, years, and neutral citations (e.g., *R. v. Smith*, 2019 SCC 65)
- **Short-form references**: Abbreviated mentions (e.g., "in *Smith*", "the *Charter*")
- **Co-references**: Multiple mentions referring to the same legal authority
- **Fragment citations**: References to specific sections, paragraphs, or provisions

Accurate extraction and resolution of these references is critical for:
- Legal information retrieval systems
- Automated citation network analysis
- Precedent recommendation systems
- Legal research assistance tools

### Research Gap

While citation extraction has been studied in scientific literature, legal citations present unique challenges:
- **Ambiguous short forms**: "the Court" may refer to different judicial bodies depending on context
- **Multi-level structure**: Citations may include titles, references, and fragments with varying granularity
- **Domain expertise required**: Understanding legal citation conventions and hierarchy
- **No existing benchmarks**: Current legal NLP datasets lack comprehensive annotation for both extraction and resolution tasks

---

## Annotation Methodology

Our annotation pipeline combines LLM capabilities with human expertise in a three-stage process:

```
┌─────────────────────────┐
│  1. LLM Pre-Annotation  │
│  (GPT-4 with prompts)   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  2. Co-reference         │
│     Disambiguation       │
│  (LLM-based linking)    │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  3. Expert Review        │
│  (Legal annotators)     │
└───────────┬─────────────┘
            │
            ▼
    Final Gold Standard
```

### Stage 1: LLM-Based Extraction

We use GPT-4 with carefully engineered prompts to identify legal authorities. The extraction process:
- Processes documents in semantically-aware chunks
- Identifies three main authority types: **legislation**, **decisions** (case law), and **secondary sources**
- Extracts hierarchical components: titles, references, and document fragments
- Assigns attributes (e.g., document IDs, fragment identifiers, title types)

### Stage 2: Co-reference Resolution

LLMs perform entity linking to resolve:
- Multiple mentions of the same authority across the document
- Short-form references to previously mentioned full citations
- Implicit references requiring contextual understanding

### Stage 3: Expert Annotation Review

Legal experts validate and correct LLM annotations, ensuring:
- Accuracy of extracted spans
- Correctness of authority classification
- Proper co-reference linking
- Adherence to legal citation conventions

---

## Evaluation Framework

Our research investigates four key research questions:

### Research Questions

**RQ1: Human Annotation Performance**  
*How reliable and efficient are human annotators when working independently?*

- **Efficiency**: Time per token, time per labeled mention, annotation throughput
- **Quality & Agreement**: Span-level (Level 1) and attribute-level (Level 2) inter-annotator agreement
- **Reliability**: Intra-annotator consistency, temporal drift, label distribution stability

**RQ2: LLM Annotation Performance**  
*To what extent can LLMs reproduce human annotations, and how stable are their outputs?*

- **Efficiency & Cost**: Token usage (input/output), monetary cost per document
- **Quality**: Agreement with human annotations, precision, recall, F1 scores
- **Determinism**: Cross-run agreement on identical documents (addressing LLM non-determinism)

**RQ3: Comparative Analysis**  
*How do humans and LLMs compare in quality, consistency, and cost?*

- **Efficiency**: Human time cost vs. LLM token cost, cost per labeled mention
- **Qualitative Differences**: Systematic omissions, over-generation, class-specific divergences

**RQ4: Collaborative Annotation**  
*Does a collaborative Human-LLM workflow improve quality-cost trade-offs?*

- **End-to-End Quality**: Agreement with human-only reference annotations
- **Efficiency Gains**: Review time vs. full manual annotation, combined human-LLM costs
- **Error Reduction**: Before-and-after correction analysis

### Evaluation Metrics

#### Span-Level Metrics (Level 1)
- **Exact match**: Full boundary agreement
- **Partial match**: Overlapping spans with different boundaries
- **Precision, Recall, F1**: Standard information extraction metrics

#### Attribute-Level Metrics (Level 2)
- **Tag accuracy**: Correct label assignment given correct span
- **Attribute accuracy**: Correctness of document IDs, fragment types, etc.
- **Co-reference F1**: Linking accuracy for entity resolution

#### Agreement Metrics
- **Inter-annotator agreement (IAA)**: Cohen's kappa, Krippendorff's alpha
- **Intra-annotator consistency**: Stability across repeated annotations
- **LLM determinism**: Agreement between multiple LLM runs

#### Efficiency Metrics
- **Time**: Seconds per document, seconds per token, seconds per mention
- **Cost**: API costs for LLMs, hourly rates for human annotators
- **Throughput**: Documents per hour, mentions per hour

---

## Repository Structure

```
LeREaD_annotation_process/
│
├── llm_based_annotation/          # Core LLM annotation pipeline
│   ├── main.py                     # Main entry point
│   ├── models.py                   # LLM model configurations
│   ├── process_chunks.py           # Chunk processing orchestration
│   ├── utils/                      # Utility modules
│   │   ├── chunker_utils.py        # Document chunking strategies
│   │   ├── prompt_utils.py         # Prompt construction
│   │   ├── few_shot_utils.py       # Few-shot example management
│   │   ├── processing_utils.py     # Token processing & normalization
│   │   ├── verification_utils.py   # Output validation
│   │   ├── html_utils.py           # HTML annotation handling
│   │   └── prompts/                # Prompt templates
│   ├── main_label_extraction.ipynb             # Extraction workflow
│   ├── main_sublabel_extraction.ipynb          # Hierarchical annotation
│   ├── main_label_extraction_full_doc.ipynb    # Full document processing
│   └── README.md                   # Technical documentation
│
├── annotation_utils/               # Annotation quality & analysis tools
│   ├── cleaner.py                  # HTML cleaning utilities
│   ├── IAA.py                      # Inter-annotator agreement computation
│   ├── add_label_tree.ipynb        # Label hierarchy visualization
│   └── IAA_further.ipynb           # Advanced IAA analysis
│
├── data/                           # Annotated legal documents
│   ├── Documents_Annotés/          # Gold standard annotations
│   │   ├── EG/                     # Annotator EG's work
│   │   ├── GL/                     # Annotator GL's work
│   │   ├── VP/                     # Annotator VP's work
│   │   ├── llm/                    # LLM-generated annotations
│   │   └── Comparatifs_annotations/ # IAA comparison sets
│   └── test/                       # Development test files
│
├── ressources/                     # Annotation guidelines
│   ├── label_scheme.json           # JSON label schema
│   └── label schemes.html          # Human-readable guidelines
│
└── fix_parents.py                  # HTML structure repair utility
```

---

## Label Scheme

Our annotation scheme captures three main authority types with hierarchical structure:

### 1. Legislation
References to statutes, regulations, and constitutional documents.

**Components**:
- **Title**: Official name or alias (e.g., *Canadian Charter of Rights and Freedoms*)
- **Reference**: Citation identifiers (e.g., Part I of the Constitution Act, 1982)
- **Fragment**: Specific provisions (e.g., section 15, subsection 2(a))

**Attributes**:
- `docid`: Unique document identifier
- `uri`: Persistent link to legislation
- `titletype`: "official" or "alias"
- `fragmentid`: Normalized fragment reference

### 2. Decisions (Case Law)
References to prior court decisions.

**Components**:
- **Title**: Case name (e.g., *R. v. Oakes*)
- **Reference**: Neutral citation (e.g., [1986] 1 SCR 103)
- **Fragment**: Specific paragraphs or sections (e.g., para. 64)

**Attributes**:
- `docid`: CanLII or other citation ID
- `uri`: Link to decision database
- `fragmentid`: Paragraph or page reference

### 3. Secondary Sources
References to legal scholarship, dictionaries, and treatises.

**Components**:
- **Title**: Publication name
- **Reference**: Publication details (volume, year, page)
- **Fragment**: Specific pages or sections

**Attributes**:
- `docid`: Unique identifier (if available)
- `fragmentid`: Page or section reference

### Co-reference Linking
All mentions of the same legal authority share a common `docid` for resolution.

---

## Getting Started

### Prerequisites

```bash
Python >= 3.8
OpenAI API key (for GPT models)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/LeREaD_annotation_process.git
cd LeREaD_annotation_process

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Set up OpenAI API credentials:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

2. Configure label scheme in `ressources/label_scheme.json`

3. Prepare input documents in HTML format with appropriate structure

### Usage

#### Full Document Annotation Pipeline

```python
from llm_based_annotation.models import GPTModel
from llm_based_annotation.process_chunks import process_chunks

# Initialize model
model = GPTModel(model_name="gpt-4", temperature=0.1)

# Configure annotation settings
label_config = {
    "use_simplified": True,
    "keep_attributes": ["labelname", "docid", "fragmentid"],
    "switch_type": True
}

# Process document
processed_chunks = process_chunks(
    model=model,
    token_chunks=document_chunks,
    process_prompt_path="utils/prompts/extraction_prompt.txt",
    label_config=label_config,
    few_shot_examples=examples,
    allowed_labels=["legislation", "decision", "secondary sources"],
    output_dir="./output",
    filename="document_name"
)
```

See [llm_based_annotation/README.md](llm_based_annotation/README.md) for detailed technical documentation.

#### Computing Inter-Annotator Agreement

```python
from annotation_utils.IAA import compute_iaa

# Compare two annotations
iaa_scores = compute_iaa(
    annotator1_file="data/Documents_Annotés/EG/2019SCC65_annotated_EG.html",
    annotator2_file="data/Documents_Annotés/VP/2019SCC65_annotated_VP.html",
    level="span"  # or "attribute" for Level 2 agreement
)

print(f"Cohen's Kappa: {iaa_scores['kappa']:.3f}")
print(f"F1 Score: {iaa_scores['f1']:.3f}")
```

---

## Prompt Engineering Approach

Our prompt engineering strategy emphasizes:

### 1. Task Decomposition
- **Phase 1**: Extract main authority types (legislation, decisions, secondary sources)
- **Phase 2**: Annotate hierarchical components (titles, references, fragments)
- **Phase 3**: Link co-references across the document

### 2. Few-Shot Learning
- Provide 3-5 representative examples per authority type
- Include edge cases (partial citations, ambiguous references)
- Balance examples across different citation styles

### 3. Structured Output
- Enforce XML/HTML-based annotation format
- Include validation markers (`<start>` and `<end>` tags)
- Specify attribute requirements explicitly

### 4. Verification Pipeline
```
Raw LLM Output → Post-Processing → Error Correction → Verification → Fallback (if needed)
```

- **Hallucination Check**: Ensure no text was added or removed
- **Consistency Check**: Validate balanced tags and proper nesting
- **Label Scheme Check**: Verify labels conform to schema
- **Fallback Strategy**: Retry with alternative prompts on failures

See [llm_based_annotation/utils/prompts/](llm_based_annotation/utils/prompts/) for prompt templates.

---

## Research Status

This project is part of ongoing master's research at [Mila - Quebec AI Institute](https://mila.quebec/) investigating human-AI collaboration in specialized annotation tasks. The work builds on recent advances in large language models for information extraction while addressing the unique challenges of legal text processing.

### Current Progress

- ✅ Label scheme design and annotation guidelines development
- ✅ LLM-based annotation pipeline with prompt engineering
- ✅ Verification and quality control mechanisms
- 🔄 Gold standard dataset creation (in progress)
- 🔄 Human vs. LLM comparative evaluation (in progress)
- 📅 Collaborative annotation workflow evaluation (planned)
- 📅 Full benchmark release (planned)

### Preliminary Findings

Early experiments demonstrate:
- **LLM Capability**: GPT-4 can identify legal authorities with reasonable accuracy (F1 > 0.75 for main labels)
- **Efficiency**: LLM pre-annotation reduces human review time by 40-60% compared to manual annotation from scratch
- **Challenges**: Short-form references and fragment extraction remain difficult for both LLMs and humans
- **Co-reference Resolution**: LLMs show promise but require careful prompt design and post-processing

### Future Directions

The research is currently under development, and findings are subject to refinement as we continue to explore:
- Additional LLM models (Claude, Llama, specialized legal LMs)
- Ensemble and fusion strategies for multi-model annotation
- Active learning approaches for selective human review
- Generalization to other legal systems and jurisdictions (U.S., U.K., civil law systems)
- Integration with downstream applications (legal search, citation network analysis)

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{leread_annotation_2026,
  title={Legal Authority Extraction and Resolution: An LLM-Assisted Annotation Framework},
  author={[Your Name]},
  year={2026},
  institution={Mila - Quebec AI Institute},
  note={Master's thesis research in progress}
}
```

---

## Contributing

This is an active research project. Contributions, suggestions, and collaboration inquiries are welcome:

- **Issues**: Report bugs or suggest features via GitHub Issues
- **Collaboration**: Contact [your email] for research collaboration opportunities
- **Data Contributions**: We welcome additional annotated legal documents following our schema

---

## License

This project is currently under research license. Data and code are provided for academic research purposes. Please contact the author for commercial use inquiries.

---

## Acknowledgments

- **Mila - Quebec AI Institute** for research support and infrastructure
- **Legal annotators** (EG, GL, VP) for their expertise and dedication
- **OpenAI** for GPT-4 API access
- **CanLII** for providing open access to Canadian legal decisions

---

## Contact

For questions about this research:

- **GitHub Issues**: For technical questions and bug reports
- **Email**: [your.email@mila.quebec]
- **Lab Website**: [Your research group's website]

---

**Last Updated**: February 2026  
**Research Institution**: Mila - Quebec AI Institute  
**Degree Program**: Master's in Computer Science (Machine Learning)
