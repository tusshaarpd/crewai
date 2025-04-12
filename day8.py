from crewai import Agent, Task, Crew, Process
from crewai.llms import OpenAI
from dotenv import load_dotenv
import os

# Load your .env file (make sure OPENAI_API_KEY is set)
load_dotenv()

# Initialize GPT-4o with browsing capabilities
llm = OpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)

# ------------------- AGENTS -------------------

# 1. Company Analyzer
company_analyzer = Agent(
    role="Company Analyzer",
    goal="Break down the core business of the company and its financial profile",
    backstory="A financial analyst who understands company fundamentals and performance.",
    llm=llm,
    verbose=True
)

# 2. Web Researcher with GPT-4o Browsing
web_researcher = Agent(
    role="Market Intelligence Agent",
    goal="Search the web for current financial news, analyst ratings, and quarterly performance for the company",
    backstory="A market expert who uses web search to gather real-time financial insights.",
    llm=llm,
    verbose=True
)

# 3. Investment Advisor
investment_advisor = Agent(
    role="Investment Advisor",
    goal="Give a clear and rational investment decision using all available info",
    backstory="An expert advisor that helps retail investors with actionable insights.",
    llm=llm,
    verbose=True
)

# ------------------- TASKS -------------------

# Step 1: Company Analysis
task_analyze = Task(
    description=(
        "Analyze Infosys Ltd (India). Cover its business model, products/services, "
        "industry positioning, and financial highlights (revenue, net profit, margins, etc)."
    ),
    agent=company_analyzer,
    expected_output="Concise company overview in bullet points."
)

# Step 2: Browse for financial data and news
task_web = Task(
    description=(
        "Search the web for the latest financial performance, analyst reports, earnings call summaries, and stock outlook for Infosys Ltd. "
        "Summarize findings from credible sources (news, finance blogs, earnings releases)."
    ),
    agent=web_researcher,
    expected_output="Short report summarizing web findings, including any notable analyst views or warnings."
)

# Step 3: Final Recommendation
task_recommend = Task(
    description=(
        "Based on the company analysis and web research, give a final investment recommendation for Infosys Ltd: Buy, Sell, or Hold. "
        "Justify using concrete reasoning. Format as a short note for a retail investor."
    ),
    agent=investment_advisor,
    expected_output="Investment decision with a 3-5 sentence explanation."
)

# ------------------- CREW -------------------

crew = Crew(
    agents=[company_analyzer, web_researcher, investment_advisor],
    tasks=[task_analyze, task_web, task_recommend],
    process=Process.sequential,
    verbose=True
)

# Run the workflow
final_result = crew.kickoff()

print("\n📈 Final Output:\n")
print(final_result)
