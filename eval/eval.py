import os
import numpy as np
import sys
from datasets import Dataset
from utils import get_openai_api_key
from main import rag_pipeline
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_relevancy

# Set OPENAI_API_KEY environment variable
os.environ['OPENAI_API_KEY'] = get_openai_api_key()



def ragas_eval(question):
    """Evaluates the RAG response against predefined metrics."""
    try:
        context, response = rag_pipeline(question)

        # Format context
        context = [str(c)[2:-4] for c in context]

        response_dataset = Dataset.from_dict({
            "question": [question],
            "answer": [response],
            "contexts": [context],
        })
        # RAGAS evaluation step
        metrics_to_evaluate = [faithfulness, answer_relevancy, context_relevancy] #for more info and more metrics see RAGAS documentation on langchain
        evaluation_results = evaluate(response_dataset, metrics_to_evaluate, raise_exceptions=False)
        return evaluation_results, response, context
    except Exception as e:
        print(f"Ragas evaluation failed: {e}")
        return [], [], []


if __name__ == "__eval__":

    args = sys.argv[1:]
    if not args:
        print("Usage: python main.py <question>")
        sys.exit(1)

    question = args[0]
    results, response, context = ragas_eval(question)
    print("Evaluation Results:", results)
    print("Response:", response)
    print("Context:", context)