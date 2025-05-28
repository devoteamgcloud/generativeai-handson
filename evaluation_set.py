from langchain_google_vertexai.chat_models import ChatVertexAI
from langchain_google_vertexai.embeddings import VertexAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator
import pandas as pd

from utils.documents import get_documents

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
generator_embeddings = LangchainEmbeddingsWrapper(embeddings)

generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings,
)

docs = get_documents()
total_docs = len(docs)

test_set = []
testset_size = min(50, len(docs))
batch_size = 50
for i in range(0, total_docs, batch_size):
    batch_docs = docs[i : i + batch_size]
    batch_doc_count = len(batch_docs)

    batch_testset_size = round((batch_doc_count / total_docs) * testset_size)
    batch_testset_size = max(1, batch_testset_size)

    print(
        f"Processing batch {i // batch_size + 1} with {batch_doc_count} documents"
    )
    print(f"Generating {batch_testset_size} test set items for this batch")

    try:
        dataset = generator.generate_with_langchain_docs(
            documents=batch_docs,
            testset_size=batch_testset_size,
            run_config=RunConfig(max_workers=8),
            with_debugging_logs=True,
        )
        test_set_batch = dataset.to_list()
        test_set.extend(test_set_batch)

        print(
            f"Batch {i // batch_size + 1} processed,"
            f" current test set size: {len(test_set_batch)}"
        )
    except ValueError as e:
        print(f"--- ValueError: {e} --- skip the batch")
        continue

print(f"Total test set size: {len(test_set)}")
test_set = pd.DataFrame(test_set)
print(f"Before duplication, total size : {len(test_set)}")
test_set.drop_duplicates(subset=["user_input"], inplace=True)
test_set.drop_duplicates(subset=["reference"], inplace=True)
test_set.to_csv("evaluation_dataset.csv", index=False)
