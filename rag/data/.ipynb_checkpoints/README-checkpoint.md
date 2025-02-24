# Contextual keywords generation for RAG

Problem: Independent chunking in traditional RAG systems leads to the loss of contextual information between chunks. This makes it difficult for LLMs to retrieve relevant data when context (e.g., the subject or entity being discussed) is not explicitly repeated within individual chunks.

Solution: Generate keywords for each chunk to fulfill missing contextual information. These keywords (e.g., "BMW, X5, pricing") enrich the chunk with necessary context, ensuring better retrieval accuracy. By embedding this enriched metadata, the system bridges gaps between related chunks, enabling effective query matching and accurate answer generation.

This article explains benefits of contextual chunking.

Note This method does not require calling LLM for each chunk separately, which makes it efficient.
