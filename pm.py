import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",  # or "gpt-4o"
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY")
)

# ---------- AGENTS ----------

problem_analyzer = Agent(
    role="Problem Statement Analyzer",
    goal="Deeply understand and break down product problem statements",
    backstory="An experienced product manager who dissects vague product problems and extracts user pain points, business goals, and constraints.",
    llm=llm,
    verbose=True
)

prd_creator = Agent(
    role="PRD Creator",
    goal="Create a structured Product Requirements Document based on problem analysis",
    backstory="A senior product strategist who writes clear, concise, and detailed PRDs for engineering teams and stakeholders.",
    llm=llm,
    verbose=True
)

user_story_creator = Agent(
    role="User Story Creator",
    goal="Write detailed user stories with acceptance criteria and priorities based on the PRD",
    backstory="An agile expert who converts PRDs into user-centric stories for developers, including edge cases and priorities.",
    llm=llm,
    verbose=True
)

# ---------- STREAMLIT UI ----------

st.title("🧠 PM CrewAI: From Problem Statement to PRD & User Stories")
user_input = st.text_area("✍️ Enter your product problem statement:")

if st.button("🚀 Generate PRD & User Stories") and user_input:
    
    # Step 1: Analyze Problem Statement
    task_analysis = Task(
        description=f"Analyze the following product problem: '''{user_input}'''. Extract goals, user pain points, success metrics, constraints, and scope.",
        agent=problem_analyzer,
        expected_output="A structured problem analysis in bullet points."
    )

    # Step 2: Create PRD
    task_prd = Task(
        description="Based on the problem analysis, generate a full PRD including Overview, Goals, Personas, Requirements, KPIs, and Timeline in markdown format.",
        agent=prd_creator,
        expected_output="A markdown-based PRD document."
    )

    # Step 3: Create User Stories
    task_stories = Task(
        description="Use the PRD to write prioritized user stories in markdown format. Each story should follow: As a [user], I want to [do something] so that [benefit]. Include acceptance criteria and priority for each.",
        agent=user_story_creator,
        expected_output="List of user stories in markdown format with acceptance criteria and priorities."
    )

    # Crew
    crew = Crew(
        agents=[problem_analyzer, prd_creator, user_story_creator],
        tasks=[task_analysis, task_prd, task_stories],
        process=Process.sequential,
        verbose=True
    )

    # Run
    result = crew.kickoff()

    # Output
    st.subheader("📄 Generated PRD & User Stories")
    st.markdown(result)
