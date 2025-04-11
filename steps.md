# Using CrewAI with Anaconda, Jupyter Notebook, and Ollama

This guide outlines the step-by-step process to set up and use CrewAI with Anaconda, Jupyter Notebook, and Ollama for running local Language Models (LLMs).

## Step 1: Set Up Your Anaconda Environment

1.  **Open Anaconda Prompt:** Launch the Anaconda Prompt from your Start Menu.
2.  **Create a New Environment (Recommended):**
    ```bash
    conda create -n crewai_ollama python=3.10
    ```
    *(Adjust Python version as needed: >= 3.10 and < 3.13)*
3.  **Activate the Environment:**
    ```bash
    conda activate crewai_ollama
    ```

## Step 2: Install Necessary Libraries

1.  **Install CrewAI:**
    ```bash
    pip install crewai
    ```
    *(Optional: Install with extra tools)*
    ```bash
    pip install 'crewai[tools]'
    ```
2.  **Install Ollama Python Library:**
    ```bash
    pip install ollama
    ```
3.  **Install Jupyter Notebook:**
    ```bash
    conda install -c conda-forge notebook
    # OR
    pip install notebook
    ```
4.  **Install `ipykernel` (for kernel selection in Jupyter):**
    ```bash
    conda install ipykernel
    python -m ipykernel install --user --name=crewai_ollama
    ```

## Step 3: Set Up and Run Ollama

1.  **Install Ollama:** Follow the instructions on the official Ollama GitHub: [https://github.com/ollama/ollama](https://github.com/ollama/ollama)
2.  **Run Ollama Server:** Open a new terminal and start the server:
    ```bash
    ollama serve
    ```
    *(Keep this terminal running)*
3.  **Pull a Language Model:** In the Ollama terminal, pull a model (e.g., Llama 2 or another model from `ollama list`):
    ```bash
    ollama pull llama2
    ```
    *(Replace `llama2` with your desired model name)*
4.  **List Pulled Models (Optional):**
    ```bash
    ollama list
    ```

## Step 4: Create and Run Your CrewAI Script in Jupyter Notebook

1.  **Launch Jupyter Notebook:** In your Anaconda Prompt (with `crewai_ollama` activated):
    ```bash
    jupyter notebook
    ```
2.  **Create a New Notebook:** Create a new Python 3 notebook.
3.  **Import Libraries:**
    ```python
    import crewai
    from crewai import Agent, Task, Crew, Process
    from ollama import Client
    ```
4.  **Initialize Ollama Client:**
    ```python
    ollama_client = Client(base_url='http://localhost:11434')
    ```
5.  **Define Your Agents:**
    ```python
    researcher = Agent(
        role='Senior Researcher',
        goal='Conduct thorough research on a given topic and provide well-cited findings.',
        backstory='An expert in information retrieval and analysis.',
        verbose=True,
        llm=ollama_client.chat,
        llm_config={"model": "llama2"}  # Replace "llama2" with your Ollama model
    )

    writer = Agent(
        role='Content Writer',
        goal='Write compelling and informative content based on research.',
        backstory='A skilled writer with a knack for simplifying complex information.',
        verbose=True,
        llm=ollama_client.chat,
        llm_config={"model": "llama2"}  # Use the same or a different Ollama model
    )
    ```
6.  **Define Your Tasks:**
    ```python
    research_task = Task(
        description="Research the latest advancements in AI for creative writing, focusing on techniques and applications. Identify at least three significant recent developments and their potential impact.",
        agent=researcher
    )

    write_task = Task(
        description="Based on the research findings, write a concise blog post (approximately 3 paragraphs) summarizing the latest advancements in large language models for creative writing and their potential impact. Include references to the researched developments.",
        agent=writer
    )
    ```
7.  **Create Your Crew:**
    ```python
    creative_ai_crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        verbose=2,
        process=Process.sequential
    )
    ```
8.  **Run the Crew:**
    ```python
    print("Starting the crew...")
    result = creative_ai_crew.kickoff()
    print("\nCrew's Final Result:")
    print(result)
    ```
9.  **Execute Cells:** Run the cells in your Jupyter Notebook.

**Important Notes:**

* Ensure the Ollama server is running (`ollama serve`) in a separate terminal.
* Replace `"llama2"` with the exact name of the model you have pulled in Ollama.
* Refer to the CrewAI documentation for more advanced features and configurations.
