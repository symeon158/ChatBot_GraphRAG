from neo4j_connector import neo4j_db
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_graph_data(user_query):
    cypher_query = """
    MATCH (startNode)
    WHERE startNode.name CONTAINS $user_query

    CALL apoc.path.subgraphAll(startNode, {
        maxLevel: 3,
        relationshipFilter: ">|<"
    })
    YIELD nodes, relationships

    UNWIND relationships AS r

    RETURN DISTINCT
        startNode.name      AS node_1,
        type(r)             AS relationship,
        endNode(r).name     AS node_2
    LIMIT 200
    """
    records = neo4j_db.query(cypher_query, {"user_query": user_query})

    extracted_data = [
        {
            "node_1": rec["node_1"],
            "relationship": rec["relationship"],
            "node_2": rec["node_2"]
        }
        for rec in records
    ]
    return extracted_data



def get_embedding(text):
    client = openai.OpenAI(api_key=openai.api_key)
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    embedding = response.data[0].embedding
    return embedding

def hybrid_search(user_query: str, top_k: int = 5) -> list[dict]:
    user_embedding = get_embedding(user_query)

    cypher = """
    // Step A: pick your top-k candidate nodes
    CALL {
      CALL db.index.vector.queryNodes('vector_index', $top_k, $user_embedding)
      YIELD node, score
      RETURN node, score
      UNION
      MATCH (node)
      WITH node,
           apoc.text.regreplace(node.name, "(ος|ης|ων|ση|σης|ώσεις)$", "") AS normName,
           apoc.text.regreplace($user_query, "(ος|ης|ων|ση|σης|ώσεις)$", "") AS normQuery
      WHERE apoc.text.levenshteinDistance(normName, normQuery) < 4
         OR normName CONTAINS normQuery
      RETURN node, 1.0 AS score
      UNION
      MATCH (node)-[:HAS_KEYWORD]->(k:Keyword)
      WHERE apoc.text.levenshteinDistance(k.name, $user_query) < 4
         OR k.name CONTAINS $user_query
      RETURN node, 1.0 AS score
    }
    WITH node, score
    ORDER BY score DESC
    LIMIT $top_k

    // Step B: traverse full subgraph both directions
    CALL apoc.path.subgraphAll(node, {
      maxLevel: 3,
      relationshipFilter: ">|<"
    })
    YIELD relationships

    // Step C: unwind and return every distinct edge (no final LIMIT here)
    UNWIND relationships AS r
    RETURN DISTINCT
        node.name         AS node_1,
        type(r)           AS relationship,
        endNode(r).name   AS node_2
    """

    records = neo4j_db.query(cypher, {
        "top_k": top_k,
        "user_embedding": user_embedding,
        "user_query": user_query
    })

    # Optionally clamp it in Python
    edges = [
        {"node_1": rec["node_1"],
         "relationship": rec["relationship"],
         "node_2": rec["node_2"]}
        for rec in records
    ]
    return edges[:100]  # keep top 100 edges to manage size
