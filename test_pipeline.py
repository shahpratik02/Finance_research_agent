import logging
from app.orchestrator import run_pipeline
from app.schemas import QueryInput

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # Folder is resolved relative to the project root (parent of app/), e.g. ./documents
    result = run_pipeline(
        QueryInput(
            query="What is the current state of GOOGLE stock and expected future growth?",
            output_style="memo",
            documents_folder="documents",
        )
    )
    print(result)
