import os, time, warnings
warnings.filterwarnings("ignore")
import logfire
from dotenv import load_dotenv
load_dotenv()

LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")

logfire.configure()
logfire.info('Hello, {place}', place='World')


logfire.info("notebook_started",
             part="PART 1 - BASICS",
             instructor="Divesh",
             tool="Pydantic Logfire")


with logfire.span("data_processing_simulation", dataset='llm-course', rows=1000):
    logfire.info('step_started', step=1, action='loading data')
    time.sleep(0.3)

    logfire.info('step_started', step=2, action='transforming', columns=12)
    time.sleep(0.2)

    logfire.info('step_started', step=3, action='saving results', output="/tmp/out.csv")


