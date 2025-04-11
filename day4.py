import crewai
from crewai import Agent, Task, Crew, Process
from ollama import Client

ollama_client = Client(host='http://localhost:11434')

from crewai import Agent, Task, Crew, Process

# Step 1: Define Agents

resume_analyzer = Agent(
    role='Resume Analyst',
    goal='Understand and extract key strengths and experience from a given resume.',
    backstory='An expert in resume review and career profiling.',
    verbose=True,
    llm="ollama/llama3.2",
    llm_config={"provider": "ollama"}
)

jd_analyzer = Agent(
    role='JD Analyst',
    goal='Identify key skills, experience, and qualifications from a given job description.',
    backstory='An experienced recruiter with deep understanding of job requirements.',
    verbose=True,
    llm="ollama/llama3.2",
    llm_config={"provider": "ollama"}
)

resume_writer = Agent(
    role='Resume Writer',
    goal='Rewrite the resume to better align with the job description.',
    backstory='A professional resume writer skilled at tailoring resumes for specific job roles.',
    verbose=True,
    llm="ollama/llama3.2",
    llm_config={"provider": "ollama"}
)

# Step 2: Load resume and JD from files
with open("resume.txt", "r", encoding="utf-8") as f:
    resume_content = f.read()

with open("job_description.txt", "r", encoding="utf-8") as f:
    jd_content = f.read()

# Step 3: Define Tasks

analyze_resume_task = Task(
    description=f"Analyze the following resume:\n\n{resume_content}",
    expected_output="A summary of the candidate's key skills, experiences, and strengths.",
    agent=resume_analyzer
)

analyze_jd_task = Task(
    description=f"Analyze the following job description:\n\n{jd_content}",
    expected_output="A list of required skills, experience, and role expectations.",
    agent=jd_analyzer
)

rewrite_resume_task = Task(
    description=(
        "Using the resume analysis and job description analysis, rewrite the resume to best fit the job. "
        "Highlight relevant experiences, use keywords from the JD, and ensure the format is professional."
    ),
    expected_output="A rewritten version of the resume tailored to the job description.",
    agent=resume_writer,
    input_tasks=[analyze_resume_task, analyze_jd_task]
)

# Step 4: Create Crew
resume_rewriter_crew = Crew(
    agents=[resume_analyzer, jd_analyzer, resume_writer],
    tasks=[analyze_resume_task, analyze_jd_task, rewrite_resume_task],
    verbose=True,
    process=Process.sequential  # or Process.parallel if you prefer
)

# Step 5: Run Crew
print("Kicking off Resume Rewriting Crew...")
result = resume_rewriter_crew.kickoff()
print("\nFinal Rewritten Resume:\n")
print(result)
