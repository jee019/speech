#!/usr/bin/env python3
"""Generate KOR-only NMOS & SMOS evaluation form."""

from jinja2 import FileSystemLoader, Environment
from utils import QuestionGenerator_MOS, QuestionGenerator_SMOS


def main():
    """Main function."""
    loader = FileSystemLoader(searchpath="./templates")
    env = Environment(loader=loader)
    template = env.get_template("test_single.html.jinja2")
    
    # Load KOR MOS questions
    kor_mos_questions = QuestionGenerator_MOS("filelist/NMOS_kor.csv").questions
    
    # Load KOR SMOS questions
    kor_smos_questions = QuestionGenerator_SMOS("filelist/SMOS_kor.csv").questions

    html = template.render(
        page_title="NMOS & SMOS Experiment Form (Korean)",
        language="KOR",
        language_name="Korean",
        language_name_kr="한국어",
        form_url="https://script.google.com/macros/s/AKfycbwpIHyP2N339ATjpCvaACFSYKMWKDYkbSDqsVLodigYmfz91eHF3mynhB3qBIc4tdPjYg/exec",
        mos_questions=kor_mos_questions,
        smos_questions=kor_smos_questions
    )
    
    # Write html into file
    with open("test_kor.html", "w", encoding="utf-8") as f:
        f.write(html)
        print("Done! Generated test_kor.html")
        print(f"  - KOR: {len(kor_mos_questions)} NMOS + {len(kor_smos_questions)} SMOS questions")
        print(f"  - Total: {len(kor_mos_questions) + len(kor_smos_questions)} questions")


if __name__ == "__main__":
    main()
