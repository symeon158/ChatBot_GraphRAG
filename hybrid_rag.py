from query_neo4j import get_graph_data
from simple_rag import simple_rag_query, init_chroma_collection

collection = init_chroma_collection()

def hybrid_simple_graph_search(user_query, top_k=5):
    simple_results = simple_rag_query(user_query, collection, top_k=top_k)
    simple_data = [{"node_1": chunk, "relationship": "—", "node_2": ""} for chunk in simple_results]
    graph_results = get_graph_data(user_query)
    combined = simple_data + graph_results
    return combined[:top_k]
