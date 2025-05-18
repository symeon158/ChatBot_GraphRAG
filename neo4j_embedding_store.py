import os
import openai
import numpy as np
from neo4j_connector import neo4j_db
from dotenv import load_dotenv

# Load OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_embedding(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response['data'][0]['embedding']

def store_embeddings():
    query = "MATCH (n) RETURN n.name AS text, ID(n) AS id"
    nodes = neo4j_db.query(query)

    for node in nodes:
        text = node["text"]
        node_id = node["id"]
        embedding = generate_embedding(text)

        # Convert embedding to string format for storage in Neo4j
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"

        # Store embedding in Neo4j
        update_query = """
        MATCH (n) WHERE ID(n) = $id
        SET n.embedding = $embedding
        """
        neo4j_db.query(update_query, {"id": node_id, "embedding": embedding_str})

    print("Embeddings stored successfully!")

# Run once to populate Neo4j with embeddings
store_embeddings()
