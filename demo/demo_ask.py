"""Demo: answer a question with citations. Run with `python -m demo.demo_ask`."""

from config.settings import DEMO_QUESTION
from pipeline.ask import answer_question

if __name__ == "__main__":
    print(answer_question(DEMO_QUESTION))
