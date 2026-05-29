"""
OpenSearch Vector Store -- REFERENCE / MIGRATION SKETCH

Status:
-------
This module is INTENTIONALLY commented out. It is a working
reference for the OpenSearch migration when FAISS no longer
scales (typically beyond ~1-5 million vectors, or when
multi-node HA + native pre-filtering becomes a requirement).

Why OpenSearch for production RAG:
----------------------------------
- k-NN search natively built in (Lucene/Faiss/NMSLIB engines)
- native PRE-filtering on metadata fields (critical for
  multi-tenant access control)
- hybrid (BM25 + vector) search out of the box
- AWS managed offering (Amazon OpenSearch Service) with
  IAM auth, VPC isolation, snapshots, cross-AZ replication
- horizontally scalable; same query API at 1k or 100M docs

To enable:
----------
1) pip install opensearch-py
2) Set env vars:
       OPENSEARCH_URL=https://<host>:443
       OPENSEARCH_INDEX=rag_index
       OPENSEARCH_USERNAME=...     (or use SigV4 IAM auth)
       OPENSEARCH_PASSWORD=...
3) Uncomment the implementation below.
4) Swap `from app.vectorstores.faiss_store import ...`
   in services for the OpenSearch equivalents -- the function
   signatures are intentionally identical to FAISS.
"""

# ============================================================================
# import os
#
# from langchain_community.vectorstores import OpenSearchVectorSearch
#
# from app.embeddings.embedding_generator import get_embedding_model
# from app.utils.logger import logger
#
#
# # =========================================================
# # Configuration
# # =========================================================
# OPENSEARCH_URL      = os.getenv("OPENSEARCH_URL", "https://localhost:9200")
# OPENSEARCH_INDEX    = os.getenv("OPENSEARCH_INDEX", "rag_index")
# OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME")
# OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")
#
#
# # =========================================================
# # Index Mapping (k-NN field definition)
# # =========================================================
# # OpenSearch needs an explicit mapping that declares
# # `vector_field` as type knn_vector with the right
# # dimension for your embedding model.
# # Titan v2 = 1024, OpenAI text-embedding-3-small = 1536.
# # The langchain wrapper creates this automatically the
# # first time you call from_documents().
#
#
# def _client_kwargs():
#     """Common kwargs for every OpenSearch call."""
#     return {
#         "opensearch_url": OPENSEARCH_URL,
#         "http_auth": (OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
#         "use_ssl": True,
#         "verify_certs": True,
#         "index_name": OPENSEARCH_INDEX,
#     }
#
#
# def upsert_chunks(chunks):
#     """
#     Mirror of faiss_store.upsert_chunks but writing to
#     a remote OpenSearch cluster.
#     """
#
#     embedding_model = get_embedding_model()
#
#     # from_documents() will CREATE the index with the right
#     # knn_vector mapping on first call, and APPEND on later calls.
#     vector_store = OpenSearchVectorSearch.from_documents(
#         documents=chunks,
#         embedding=embedding_model,
#         **_client_kwargs(),
#     )
#
#     logger.info(
#         "opensearch_upsert_complete",
#         index=OPENSEARCH_INDEX,
#         chunks_added=len(chunks),
#     )
#
#     return {
#         "status": "success",
#         "chunks_added": len(chunks),
#         "index_name": OPENSEARCH_INDEX,
#     }
#
#
# def similarity_search(query: str, top_k: int = 5, metadata_filter: dict = None):
#     """
#     Mirror of faiss_store.similarity_search.
#
#     IMPORTANT: OpenSearch supports NATIVE PRE-FILTERING via the
#     `efficient_filter` parameter. This is the main reason to
#     migrate from FAISS in multi-tenant systems.
#     """
#
#     embedding_model = get_embedding_model()
#
#     vector_store = OpenSearchVectorSearch(
#         embedding_function=embedding_model,
#         **_client_kwargs(),
#     )
#
#     # Build a bool/must filter clause from a simple dict.
#     # Example metadata_filter={"doc_type": "pdf", "tenant_id": "acme"}
#     # becomes:
#     #   {"bool": {"must": [
#     #       {"term": {"metadata.doc_type.keyword": "pdf"}},
#     #       {"term": {"metadata.tenant_id.keyword": "acme"}},
#     #   ]}}
#
#     efficient_filter = None
#     if metadata_filter:
#         efficient_filter = {
#             "bool": {
#                 "must": [
#                     {"term": {f"metadata.{k}.keyword": v}}
#                     for k, v in metadata_filter.items()
#                 ]
#             }
#         }
#
#     results = vector_store.similarity_search_with_score(
#         query=query,
#         k=top_k,
#         efficient_filter=efficient_filter,
#     )
#
#     return [
#         {
#             "content":    doc.page_content,
#             "metadata":   doc.metadata,
#             "similarity": float(score),
#         }
#         for doc, score in results
#     ]
