# Generative AI Hands-On

## Part 1: Build a RAG Application 🏗️

Welcome to the first part of our workshop! Here, we'll dive into building a **Retrieval-Augmented Generation (RAG)** application.

**What is RAG?** Think of it as giving a powerful language model (LLM) an "open book" test. Instead of just using what the LLM already knows, we first *find* relevant information in our own documents (the "open book") and then give that information to the LLM along with the question. This helps the LLM generate answers that are specific to our data and more up-to-date.

We'll use **LangChain** to orchestrate the process, **LangGraph** to build it as a sequence of steps, and **Google Vertex AI** for our LLM and text embedding needs.

### 1.1: Setting Up Your Environment 🛠️

Before we start coding, make sure you have the necessary tools.

1. **Install Libraries:** You'll need Python, and you can install the core libraries using uv:

    ```bash
    uv sync --all-groups
    ```

2. **Document Loader:** Create a `utils` folder and inside it, a file named `documents.py`. In this file, define a function `get_documents()`. This function's job is to load whatever documents you want to use (like PDFs, text files, etc.) and return them as a list. Each item in the list should be a `langchain_core.documents.Document` object. You MUST use LangChain's `RecursiveUrlLoader` document loader for this exercise and scrape your favorite section from Melexis' website.

3. **Google Cloud Authentication:** Ensure you're authenticated with Google Cloud and have the Vertex AI API enabled in your project (c.f. Vertex AI hands-on).

### 1.2: The Plan - Building with `get_graph` 🗺️

We will structure our code within a main function, let's call it `get_graph()`. This function will set up all the components and build our RAG application graph, returning the compiled graph ready to be used.

### 1.3: Initializing Core Components ⚙️

Inside `get_graph()`, we need to set up the brains and the "search engine" of our RAG system.

* **The LLM:** We need a language model to generate answers.
    * **Hint:** Use `ChatVertexAI` from `langchain_google_vertexai.chat_models`.
    * **Think:** Which model should you use (e.g., `gemini-2.5-flash-preview-05-20`)? What `location`? Should you set `temperature` to 0.0 for more predictable results?
* **The Embedder:** We need a way to turn text into numbers (vectors) so we can search for similar pieces of text.
    * **Hint:** Use `VertexAIEmbeddings` from `langchain_google_vertexai.embeddings`.
    * **Think:** Which embedding model (`e.g., text-embedding-005`) and `location`?
* **The Vector Store:** This is where we'll store our text chunks and their corresponding vectors for fast searching. For simplicity, we'll use one that works in memory.
    * **Hint:** Use `InMemoryVectorStore` from `langchain_core.vectorstores`.
    * **Think:** What does `InMemoryVectorStore` need when you initialize it? (It needs the *embedder*).

### 1.4: Loading and Processing Documents 📄

Now, let's get our documents ready for the RAG process.

* **Load:** Call your `get_documents()` function to load your source material.
* **Split:** LLMs have limited input windows, so we need to break our documents into smaller chunks.
    * **Hint:** Use `RecursiveCharacterTextSplitter` from `langchain_text_splitters`.
    * **Think:** What `chunk_size` (e.g., 1000 characters) and `chunk_overlap` (e.g., 200 characters) make sense? How do you call it to split your loaded `docs`?
* **Store:** Convert these chunks into vectors and save them in the vector store.
    * **Hint:** Your `vector_store` object has a method for this. Look for something like `add_documents`.

### 1.5: Designing the RAG Flow with LangGraph 🧠

LangGraph helps us define our application as a *graph* of steps.

* **The Prompt:** We need a template to structure the input for the LLM, telling it how to use the retrieved context.
    * **Hint:** Use `langchain.hub.pull`. Look for a common RAG prompt like `"rlm/rag-prompt"`.
* **The State:** LangGraph works by passing a "state" object between steps. We need to define what information our state should hold.
    * **Hint:** Use `TypedDict` from `typing_extensions`.
    * **Think:** What information needs to flow through our RAG process? We'll definitely need the `question`, the retrieved `context` (which should be a `List` of `Document` objects), and finally the `answer`.
* **The Nodes (Steps):** We need to define Python functions for each step in our RAG flow. Each function will take the current `State` as input and return a dictionary with the parts of the state it has updated.
    * **`retrieve` Node:** This function should take the `question` from the state and use the `vector_store` to find relevant documents.
        * **Hint:** Use the `similarity_search` method of your `vector_store`.
        * **Think:** What should this function return? (A dictionary: `{"context": ...}`).
    * **`generate` Node:** This function should take the `question` and the `context` from the state, use the RAG `prompt` to format them, call the `llm` to get an answer, and return the answer.
        * **Hint:** You'll need to combine the `page_content` from all documents in the `context`. Then, use the `prompt.invoke()` method, followed by `llm.invoke()`.
        * **Think:** What should this function return? (A dictionary: `{"answer": ...}`).

### 1.6: Building and Compiling the Graph 🕸️

Now, let's assemble our nodes into a working graph.

* **Initialize:** Create an instance of `StateGraph` from `langgraph.graph`, passing your `State` definition.
* **Add Nodes:** Add your `retrieve` and `generate` functions as nodes.
    * **Hint:** Use the `add_node` method, giving each node a name (e.g., "retrieve", "generate") and a reference to your function.
* **Define Edges:** Define how the state flows from one node to the next. For a simple RAG, it's a sequence.
    * **Hint:** You can use `add_sequence` for a linear flow, or `add_edge` to define connections. You need to specify the `START` point (import `START` from `langgraph.graph`).
* **Compile:** Turn your graph definition into a runnable application.
    * **Hint:** Use the `compile()` method.
* **Return:** Make sure your `get_graph()` function returns the compiled graph.

### 1.7: Running Your RAG Application 🚀

Finally, let's make the script runnable so you can test it.

* **Hint:** Use the standard Python `if __name__ == "__main__":` block.
* **Inside this block:**
    1.  Call `get_graph()` to get your compiled RAG application.
    2.  Call the `invoke()` method on your graph.
    3.  **Think:** What does `invoke` need as input? (It needs a dictionary matching your `State`, at least with the initial `question`).
    4.  Print the `answer` from the result!

---

Good luck with the coding! Don't hesitate to consult the LangChain and LangGraph documentation if you get stuck, or ask your workshop instructors for a nudge in the right direction. The goal is to understand *how* these pieces fit together by building it yourself.

## Part 2: Generate a Synthetic Dataset 🧪

Now that we have a RAG application, how do we know if it's working well? We need to test it! Manually creating a comprehensive test set (questions and expected answers) can be tedious. In this part, we'll use a library called **Ragas** to automatically generate a test set directly from our documents.

**What is Ragas?** Ragas is a framework specifically designed for evaluating RAG pipelines. One of its handy features is the ability to generate question/answer pairs from your documents, which we can then use as a "ground truth" (or close to it) for testing.

### 2.1: Setting Up Your Environment 🛠️

We need a few more libraries for this part.

1.  **Install Libraries:** If you haven't already, install `ragas` and `pandas`:
    ```bash
    uv add ragas pandas
    ```
2.  **Document Loader:** We'll use the same `utils/documents.py` and `get_documents()` function from Part 1.

### 2.2: Initializing LLM and Embeddings for Ragas ⚙️

Ragas needs its own connection to an LLM and an embedding model to generate the questions and answers.

* **LLM & Embeddings:** Just like in Part 1, you'll need to initialize `ChatVertexAI` and `VertexAIEmbeddings`. You can use the same models and settings.
* **Ragas Wrappers:** Ragas has its own way of interacting with models. We need to wrap our LangChain models so Ragas can understand them.
    * **Hint:** Look for `LangchainLLMWrapper` in `ragas.llms` and `LangchainEmbeddingsWrapper` in `ragas.embeddings`.
    * **Think:** How do you pass your initialized LangChain LLM and Embeddings into these wrappers?

### 2.3: Creating the Testset Generator 🧬

With our wrapped models ready, we can create the main Ragas tool for this task.

* **Hint:** You'll need to instantiate `TestsetGenerator` from `ragas.testset`.
* **Think:** What arguments does `TestsetGenerator` likely need? (It needs the `llm` and `embedding_model` – make sure to use your *wrapped* versions!).

### 2.4: Loading Documents 📄

As before, load your documents using your `get_documents()` function. Keep track of how many documents you've loaded (`len(docs)`).

### 2.5: Generating in Batches ➗

Generating a test set can be resource-intensive, especially with many documents. It's often safer and more manageable to process documents in smaller batches.

* **Plan:**
    * Decide on a total `testset_size` you want (e.g., 50 questions).
    * Decide on a `batch_size` (e.g., 50 documents per batch).
    * Create an empty list (e.g., `test_set`) to hold all the generated questions.
    * You'll need a loop that goes through your `docs` list in steps of `batch_size`.
* **Inside the Loop:**
    * Get the current batch of documents.
    * Calculate how many test questions to generate for *this specific batch*. This should be proportional to the total number of documents (e.g., `(batch_doc_count / total_docs) * testset_size`). Remember to round it and make sure it's at least 1.
    * Call the generator.
        * **Hint:** Use the `generate_with_langchain_docs` method of your `generator` object.
        * **Think:** What arguments does it need? You'll need to provide the `documents` (your current batch) and the `testset_size` (your calculated *batch* test set size). You might also want to look into `RunConfig` (from `ragas.run_config`) to potentially speed things up with `max_workers`.
    * **Error Handling:** Generation can sometimes fail, especially with diverse documents. It's wise to wrap the generation call in a `try...except` block (especially for `ValueError`) and print a message if a batch fails, allowing the loop to continue.
    * **Collect Results:** The generator returns a special `Testset` object. You need to convert it into a more usable format.
        * **Hint:** Look for a `.to_list()` method on the result.
        * Add the results from this batch to your main `test_set` list.

### 2.6: Cleaning and Saving Your Dataset 💾

Once the loop finishes, you'll have a list of generated questions and answers. Let's clean it up and save it.

* **Convert to DataFrame:** It's much easier to work with this data as a table.
    * **Hint:** Use `pandas.DataFrame()` to convert your `test_set` list.
* **Remove Duplicates:** Synthetic generation can sometimes create very similar or identical questions or answers. Let's remove them.
    * **Hint:** Use the `.drop_duplicates()` method on your DataFrame.
    * **Think:** Which columns should you check for duplicates? The question itself (often called `user_input` or similar by Ragas) and the ground truth answer (`reference`) are good candidates.
* **Save to CSV:** Save your cleaned dataset so you can use it in the next step.
    * **Hint:** Use the `.to_csv()` method on your DataFrame.
    * **Think:** Give it a filename (like `evaluation_dataset.csv`). Should you include the index column? (Probably not, so use `index=False`).

---

Great! If everything ran correctly, you should now have a `evaluation_dataset.csv` file filled with questions and reference answers based on your documents. In the final part, we'll use this dataset to evaluate the RAG application we built earlier.

Okay, here are the instructions for the final part of your workshop: **Evaluating the RAG Application**.

---

## Part 3: Evaluate the RAG Application 📊

We've built a RAG application (Part 1) and generated a test dataset (Part 2). Now it's time to put our RAG app to the test and see how well it performs! We'll use **Ragas** again, this time to calculate various metrics that tell us about the quality of our RAG system's answers and its retrieval process.

**Why Evaluate?** Evaluation tells us if our RAG application is **Faithful** (doesn't make things up), **Relevant** (answers the question), and if its **Context Retrieval** is effective (finds the right information). This helps us understand its strengths and weaknesses and guide improvements.

### 3.1: Prerequisites 📋

* Make sure you have your `evaluation_dataset.csv` file from Part 2.
* Ensure your RAG application code (e.g., `rag_app.py`) with the `get_graph()` function is available to be imported.
* You'll need `pandas` and `ragas` installed.

### 3.2: Loading Your Test Set 💾

The first step is to load the questions and reference answers we generated earlier.

* **Hint:** Use `pandas.read_csv()` to load your `evaluation_dataset.csv` into a DataFrame.

### 3.3: Running Your RAG App on the Test Questions 🏃‍♀️

We need to run every question from our test set through the RAG application we built in Part 1 to get its actual answers and the documents it retrieved. Since running questions one by one can be slow, we'll try to run them in a batch.

* **Create a Helper Function:** Define a function, say `get_answers(questions: list[str])`, that takes a list of question strings.
* **Inside the Function:**
    * Import your `get_graph` function from your Part 1 code.
    * Call `get_graph()` to get your compiled RAG application.
    * Your graph's `batch` method expects a list of dictionaries, not just strings. You'll need to transform your input list into `[{"question": q1}, {"question": q2}, ...]`.
    * **Hint:** Use the `graph.batch()` method. This is much faster than calling `invoke` in a loop.
    * Return the list of results from the `batch` call.
* **Get Results:** Call your new `get_answers` function, passing it the `user_input` column from your DataFrame (convert it to a list first using `.tolist()`).
* **Add to DataFrame:** The results will be a list of dictionaries (each matching your `State` from Part 1). You need to extract the `answer` and the `context` from each result and add them as new columns to your DataFrame.
    * **Think:** How do you access values in a list of dictionaries? For the `context`, Ragas expects a list of *strings* (the `page_content`), not `Document` objects. You'll need to process the context list accordingly.

### 3.4: Preparing the Data for Ragas Evaluation 🍱

Ragas needs the data in a specific structure to perform the evaluation. We need to convert our DataFrame into a `ragas.EvaluationDataset`.

* **Handle `reference_contexts`:** The `reference_contexts` column in your CSV might be stored as a string representation of a list (e.g., `"['context1', 'context2']"`). Ragas needs it as an actual Python list.
    * **Hint:** Write a small helper function `parse_reference_contexts(value)`. Inside it, try using `ast.literal_eval` (you'll need to `import ast`). Since `ast.literal_eval` can be strict, you might want a `try...except` block that falls back to `json.loads` (`import json`) or just returns the value if parsing fails. Make sure to handle cases where it might *already* be a list (if you re-run this without saving/loading).
* **Create `SingleTurnSample` Objects:** Ragas uses `SingleTurnSample` (from `ragas`) to represent each Q&A pair along with its context.
    * **Hint:** You'll need to iterate through each row of your DataFrame (using `.iterrows()` is one way) and create a `SingleTurnSample` for each.
    * **Think:** Look at the `SingleTurnSample` documentation or its signature. You'll need to map your DataFrame columns (`user_input`, `reference`, `answer`, `retrieved_contexts`) to its parameters. Remember to use your `parse_reference_contexts` function for the `reference_contexts`.
* **Create `EvaluationDataset`:** Collect all your `SingleTurnSample` objects into a list and use it to create an `EvaluationDataset` (from `ragas`).
    * **Hint:** `EvaluationDataset(samples=your_list_of_samples)`.

### 3.5: Configuring the Evaluation Metrics 📏

Now, let's choose which aspects of our RAG system we want to measure.

* **Initialize LLM/Embeddings:** Ragas needs LLM and embedding models (just like before) for some of its metrics, which use LLMs to *judge* the quality.
    * **Hint:** Initialize `ChatVertexAI` and `VertexAIEmbeddings`. You might need `LangchainLLMWrapper` for certain metrics.
* **Select Metrics:** Choose a set of metrics from `ragas.metrics`. Good starting points include:
    * `ResponseRelevancy`: Is the answer relevant to the question?
    * `Faithfulness`: Does the answer stick to the provided context?
    * `LLMContextPrecisionWithReference`: Are the retrieved contexts relevant, judged by an LLM against a reference?
    * `LLMContextRecall`: Did we retrieve all the necessary context, judged by an LLM?
    * `ContextEntityRecall`: Did we retrieve documents containing key entities from the reference answer? (May need an LLM passed in).
    * `NoiseSensitivity`: Does the RAG system's answer change significantly if noisy (irrelevant) documents are added to the context? (Needs an LLM).
    * **Think:** Create a list containing instances of these metric classes. Check if any require you to pass the `llm` during initialization.
* **Configure Run:** Set up how Ragas should perform the evaluation.
    * **Hint:** Use `RunConfig` from `ragas`. You can set `max_workers` for parallelism and maybe a `timeout`.

### 3.6: Running the Evaluation and Seeing the Results! 🎉

This is the moment of truth!

* **Call `evaluate`:** Use the main `ragas.evaluate` function.
    * **Think:** What arguments will it need? You'll need to pass your `EvaluationDataset`, your list of `metrics`, the `RunConfig`, and likely the `llm` and `embeddings` you initialized in step 3.5. You can also set a `batch_size` here to control how many evaluations run at once.
* **Print Results:** The `evaluate` function returns a dictionary (or a similar object) containing the scores for each metric. Print it out!

---

Congratulations! You've now not only built a RAG application but also systematically evaluated its performance using a synthetically generated dataset. The scores you see give you valuable insights. Low faithfulness might mean you need to adjust your prompt or LLM settings. Low context recall might mean your chunking or retrieval strategy needs a rethink. This is the starting point for iterating and improving your Generative AI solution!
