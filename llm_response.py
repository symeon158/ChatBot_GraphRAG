# llm_response.py

import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# Load environment variables from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Missing OpenAI API Key. Set it in the .env file.")

# Initialize the ChatOpenAI client (GPT-4o) with zero temperature for deterministic outputs
llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0,
    openai_api_key=openai_api_key
)

# Prompt template for structured Greek responses
_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Είσαι ένας εξειδικευμένος ψηφιακός βοηθός για τον **Εθνικό Κατάλογο Υπηρεσιών της Ελλάδας**. 
Σου δίνω πληροφορίες από τη Γραφική Βάση Γνώσης (GraphRAG) και θέλω να απαντήσεις με **δομημένο και επαγγελματικό τρόπο στα ελληνικά**.


🔹 **Ερώτηση:** {question}
🔹 **Σχετικές Πληροφορίες:**
{context}

🔹 **Απάντηση:** Δώσε μια αναλυτική, σαφή και δομημένη απάντηση στα ελληνικά. Εξήγησε τη διαδικασία, τα βήματα, και αν υπάρχουν σχετικές υπηρεσίες.
"""
)

def generate_response(user_query: str, graph_data: list[dict], mode: str = "text_only") -> str:
    """
    Γεννά μια απάντηση στα ελληνικά, χρησιμοποιώντας δεδομένα από τη βάση γνώσης.
    
    :param user_query: Το ερώτημα του χρήστη.
    :param graph_data: Λίστα από dicts με κλειδιά 'node_1', 'relationship', 'node_2' (και προαιρετικά 'score').
    :param mode: "text_only" ή "hybrid" — αλλά η μορφοποίηση παραμένει ίδια.
    :return: Το κείμενο της απάντησης από το LLM.
    """

    # Κατασκευάζουμε το context με την ίδια δομή για text-only και hybrid:
    context_lines = []
    for item in graph_data:
        node1 = item.get("node_1", "")
        rel   = item.get("relationship", "")
        node2 = item.get("node_2", "")
        context_lines.append(f"➤ **{node1}** → {rel} → **{node2}**")
    context = "\n".join(context_lines)

    # Γεμίζουμε το πρότυπο
    prompt = _prompt.format(context=context, question=user_query)

    # Καλούμε το LLM για πρόβλεψη
    response = llm.predict(prompt)
    return response