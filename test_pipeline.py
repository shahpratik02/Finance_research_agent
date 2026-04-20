import logging
from app.orchestrator import run_pipeline
from app.schemas import QueryInput

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    result = run_pipeline(QueryInput(query="What is the current state of Apple stock?", output_style="memo"))
    print(result)
