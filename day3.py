import crewai
from crewai import Agent, Task, Crew, Process
from ollama import Client


# Initialize the Ollama client with the base URL as the 'host' argument
ollama_client = Client(host='http://localhost:11434')
researcher = Agent(
    role='Senior Researcher',
    goal='Conduct thorough research on a given topic and provide well-cited findings.',
    backstory='An expert in information retrieval and analysis.',
    verbose=True,
    llm="ollama/llama3.2",  # Instead of passing a client object
    llm_config={"provider": "ollama"}
)

writer = Agent(
    role='Content Writer',
    goal='Write compelling and informative content based on research.',
    backstory='A skilled writer with a knack for simplifying complex information.',
    verbose=True,
    llm="ollama/llama3.2",  # Use the same fix
    llm_config={"provider": "ollama"}
)


research_task = Task(
    description="Gather insights on System design concepts.",
    agent=researcher,
    expected_output="A detailed report outlining key system design concepts with explanations."
)

write_task = Task(
    description="Based on the research findings, write a concise blog post (approximately 3 paragraphs) summarizing the concepts in point form.",
    agent=writer,
    expected_output="A blog post with a point-form summary of the system design concepts."
)

creative_ai_crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    verbose=True,  # Changed from 2 to True to enable verbose output
    process=Process.sequential
)

print("Starting the crew...")
result = creative_ai_crew.kickoff()
print("\nCrew's Final Result:")
print(result)
