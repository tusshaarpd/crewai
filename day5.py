import crewai
from ollama import Client
# pip install crewai pymupdf

from crewai import Agent, Task, Crew, Process
import fitz  # PyMuPDF

# --- Helper Function to Extract PDF Text ---
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


# --- Step 1: Extract Resume & JD from PDFs ---
resume_content = extract_text_from_pdf("resume.pdf")
jd_content = extract_text_from_pdf("job_description.pdf")


# --- Step 2: Define Agents using Ollama with llama3 ---
resume_analyzer = Agent(
    role='Resume Analyst',
    goal='Understand and extract key strengths and experience from a given resume.',
    backstory='An expert in resume review and career profiling.',
    verbose=True,
    llm="ollama/llama3.2:latest",
    llm_config={"provider": "ollama"}
)

jd_analyzer = Agent(
    role='JD Analyst',
    goal='Identify key skills, experience, and qualifications from a given job description.',
    backstory='An experienced recruiter with deep understanding of job requirements.',
    verbose=True,
    llm="ollama/llama3.2:latest",
    llm_config={"provider": "ollama"}
)

resume_writer = Agent(
    role='Resume Writer',
    goal='Rewrite the resume to better align with the job description.',
    backstory='A professional resume writer skilled at tailoring resumes for specific job roles.',
    verbose=True,
    llm="ollama/llama3.2:latest",
    llm_config={"provider": "ollama"}
)


# --- Step 3: Define Tasks ---
analyze_resume_task = Task(
    description=f"""Analyze the following resume and summarize key information:
{resume_content}

Provide:
- Candidate's top skills
- Experience areas and projects
- Tech/tools mentioned
- Strengths to highlight""",
    expected_output="A structured summary of skills, experiences, tools, and strengths.",
    agent=resume_analyzer
)

analyze_jd_task = Task(
    description=f"""Analyze the following job description and extract:
{jd_content}

Provide:
- Must-have and nice-to-have skills
- Required experience levels
- Keywords used
- Role expectations and deliverables""",
    expected_output="Structured breakdown of job requirements and expectations.",
    agent=jd_analyzer
)

rewrite_resume_task = Task(
    description="Using the resume and JD analysis, rewrite the resume to match the job. Include relevant skills, use JD keywords, and ensure clarity and professionalism.",
    expected_output="A rewritten resume aligned with the job description.",
    agent=resume_writer,
    input_tasks=[analyze_resume_task, analyze_jd_task]
)


# --- Step 4: Create Crew ---
resume_rewriter_crew = Crew(
    agents=[resume_analyzer, jd_analyzer, resume_writer],
    tasks=[analyze_resume_task, analyze_jd_task, rewrite_resume_task],
    verbose=True,
    process=Process.sequential  # Maintain ordered flow
)


# --- Step 5: Run Crew ---
print("🔁 Running the Resume Rewriter Crew...\n")
result = resume_rewriter_crew.kickoff()
print("\n✅ Final Rewritten Resume:\n")
print(result)
