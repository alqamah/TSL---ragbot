# Project 1: Upgrade the Existing SOP RAGBot

> **Priority:** Highest

The existing RAGBot already demonstrates:

- FastAPI
- Qdrant
- Gemini embeddings
- Multilingual question answering
- Industrial SOP retrieval
- Source citations
- Basic telemetry

Develop it into a credible enterprise application by implementing the following capabilities.

## Features to Implement

### 1. Document Metadata and Versioning

Track the following metadata for every document:

- Department
- Equipment identifier
- Equipment category
- Document owner
- Document revision
- Approval status
- Effective date
- Expiration date

The assistant should prefer current, approved procedures and avoid outdated documents.

### 2. Hybrid Search

Combine:

- Semantic vector search
- Keyword or sparse search
- Equipment identifiers
- SOP numbers
- Technical terminology
- Maintenance abbreviations

Industrial queries frequently contain exact identifiers that pure vector search can miss. Qdrant officially supports combining semantic and lexical retrieval, including hybrid queries and reranking.

**References:** Qdrant: *Hybrid Search*; Qdrant: *Hybrid Search with Reranking*

### 3. Better Citations

Show the following with each answer where applicable:

- Document name
- Revision number
- Page number or section
- Spreadsheet sheet name and row reference
- Retrieval confidence

### 4. Role-Based Document Access

Support roles such as:

- Maintenance assistant
- Supervisor
- Department manager
- Safety officer
- Administrator

Apply access restrictions **before** document retrieval, not only after an answer is generated.

### 5. Safe Handling of Unsupported Questions

For unsupported questions, the system should clearly state:

> I could not find an approved source for this question. Please consult the responsible supervisor or the latest approved procedure.

Do not allow the model to invent maintenance instructions.

### 6. Multilingual Support

Support:

- English
- Hindi
- Hinglish
- Common departmental abbreviations
- Equipment aliases

### 7. User Feedback

Allow users to report:

- Helpful answer
- Incorrect source
- Missing document
- Outdated information
- Unsafe recommendation

### 8. Evaluation Dashboard

Track:

- Retrieval success
- Citation accuracy
- Answer faithfulness
- Unanswerable-question handling
- Response time
- Cost per query
- User feedback

Ragas documents standard metrics including context precision, context recall, response relevancy, and faithfulness.

**Reference:** Ragas: *Available Evaluation Metrics*

### 9. Security Testing

Test scenarios such as:

- A document containing instructions that attempt to override the system prompt
- A user asking for restricted departmental documents
- Uploaded files containing malicious instructions
- Questions that request unsafe operational decisions

OWASP identifies prompt injection, including attacks introduced through external documents, as a significant risk for LLM applications.

**Reference:** OWASP: *Prompt Injection*

## Why This Project Matters

This project demonstrates enterprise retrieval, security, evaluation, and practical industrial use cases.
