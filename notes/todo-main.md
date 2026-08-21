Project 1: Upgrade the existing SOP RAGBot
This should be your highest-priority project.
Your existing RAGBot already demonstrates:
• FastAPI.
• Qdrant.
• Gemini embeddings.
• Multilingual question answering.
• Industrial SOP retrieval.
• Source citations.
• Basic telemetry.
Now develop it into a credible enterprise application.
Features to implement:
1. Document metadata and versioning
Track:
• Department.
• Equipment identifier.
• Equipment category.
• Document owner.
• Document revision.
• Approval status.
• Effective date.
• Expiration date.
The assistant should prefer current, approved procedures and avoid outdated documents.
2. Hybrid search
Combine:
• Semantic vector search.
• Keyword or sparse search.
• Equipment identifiers.
• SOP numbers.
• Technical terminology.
• Maintenance abbreviations.
This matters because industrial queries frequently contain exact identifiers that pure vector search can miss.
Qdrant officially supports combining semantic and lexical retrieval, including hybrid queries and reranking.
Qdrant: Hybrid Search, Qdrant: Hybrid Search with Reranking
3. Better citations
Show:
• Document name.
• Revision number.
• Page number or section.
• Spreadsheet sheet name and row reference when relevant.
• Retrieval confidence.
4. Role-based document access
Example roles:
• Maintenance assistant.
• Supervisor.
3
• Department manager.
• Safety officer.
• Administrator.
Apply access restrictions before document retrieval, not just after generating an answer.
5. Safe handling of unsupported questions
The system should clearly say:
I could not find an approved source for this question. Please consult the responsible supervisor or the latest
approved procedure.
Do not let the model invent maintenance instructions.
6. Multilingual support
Support:
• English.
• Hindi.
• Hinglish.
• Common departmental abbreviations.
• Equipment aliases.
7. User feedback
Allow users to indicate:
• Helpful answer.
• Incorrect source.
• Missing document.
• Outdated information.
• Unsafe recommendation.
8. Evaluation dashboard
Track:
• Retrieval success.
• Citation accuracy.
• Answer faithfulness.
• Unanswerable-question handling.
• Response time.
• Cost per query.
• User feedback.
Ragas documents standard metrics including context precision, context recall, response relevancy, and
faithfulness. Ragas: Available Evaluation Metrics
9. Security testing
Test scenarios such as:
• A document containing instructions that attempt to override the system prompt.
• A user asking for restricted departmental documents.
• Uploaded files containing malicious instructions.
• Questions that request unsafe operational decisions.
4
OWASP identifies prompt injection, including attacks introduced through external documents, as a significant
risk for LLM applications. OWASP: Prompt Injection
Why this project matters: It shows that you understand enterprise retrieval, security, evaluation, and practical
industrial use cases.