import pandas as pd
import json
from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
    ContextEntityRecall,
    NoiseSensitivity,
)
from langchain_google_vertexai.chat_models import ChatVertexAI
from langchain_google_vertexai.embeddings import VertexAIEmbeddings
from ragas.llms import LangchainLLMWrapper


from rag import get_graph
import ast


def get_answers(questions: list[str]) -> list[str]:
    """Get answers to a list of questions using RAG."""
    graph = get_graph()
    questions = [
        {"question": question}
        for question in questions
    ]
    print("Batching started...")
    answers = graph.batch(
        questions,
    )
    print("Batching finished.")
    return answers

df = pd.read_csv("evaluation_dataset.csv")
answers = get_answers(df["user_input"].tolist())
df["answer"] = [
    answer["answer"]
    for answer in answers
]
df["retrieved_contexts"] = [
    [
        str(doc.page_content)
        for doc in answer["context"]
    ]
    for answer in answers
]

print(df.describe(include="all"))

def parse_reference_contexts(value):
    if isinstance(value, list):
        return value
    try:
        # Try to parse using ast.literal_eval for strings like "['item1', 'item2']"
        return ast.literal_eval(value)
    except Exception:
        # Fallback to json.loads if possible
        try:
            return json.loads(value)
        except Exception:
            # Return as is if parsing fails
            return value

samples = [
    SingleTurnSample(
        user_input=row["user_input"],
        reference=row["reference"],
        response=row["answer"],
        retrieved_contexts=row["retrieved_contexts"],
        reference_contexts=parse_reference_contexts(row["reference_contexts"]),
    )
    for _, row in df.iterrows()
]
eval_dataset = EvaluationDataset(samples=samples)

llm = ChatVertexAI(
    model_name="gemini-2.5-flash-preview-05-20",
    location="us-central1",
    temperature=0.0,
)
generator_llm = LangchainLLMWrapper(
    llm,
)
embeddings = VertexAIEmbeddings(
    model_name="text-embedding-005",
    location="us-central1",
)
metrics = [
    ResponseRelevancy(),
    Faithfulness(),
    LLMContextPrecisionWithReference(),
    ContextEntityRecall(llm=generator_llm),
    NoiseSensitivity(llm=generator_llm),
    LLMContextRecall(),
]
max_workers = 8
run_config = RunConfig(max_workers=max_workers, timeout=60)

evaluation_batch_size = 32

evaluation_results = evaluate(
    dataset=eval_dataset,
    metrics=metrics,
    run_config=run_config,
    llm=llm,
    embeddings=embeddings,
    batch_size=evaluation_batch_size,
)
print(evaluation_results)
