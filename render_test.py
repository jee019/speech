#!/usr/bin/env python3
"""Generate combined NMOS & SMOS forms for human evaluation (ENG + KOR)."""

from jinja2 import FileSystemLoader, Environment
from utils import QuestionGenerator_MOS, QuestionGenerator_SMOS


def main():
    """Main function."""
    loader = FileSystemLoader(searchpath="./templates")
    env = Environment(loader=loader)
    template = env.get_template("test.html.jinja2")
    
    # Load ENG MOS questions
    eng_mos_questions = QuestionGenerator_MOS("filelist/NMOS_eng.csv").questions
    
    # Load ENG SMOS questions
    eng_smos_questions = QuestionGenerator_SMOS("filelist/SMOS_eng.csv").questions
    
    # Load KOR MOS questions
    kor_mos_questions = QuestionGenerator_MOS("filelist/NMOS_kor.csv").questions
    
    # Load KOR SMOS questions
    kor_smos_questions = QuestionGenerator_SMOS("filelist/SMOS_kor.csv").questions

    html = template.render(
        page_title="NMOS & SMOS Experiment Form (ENG + KOR)",
        form_url="https://script.google.com/macros/s/AKfycbz88pG3Usw7jdtNWh9pUWnfPR855SESSYZFGyaYbPv006CSZplhdrc2v2wwhjakF8mkjg/exec",  # Google Apps Script URL
        eng_mos_questions=eng_mos_questions,
        eng_smos_questions=eng_smos_questions,
        kor_mos_questions=kor_mos_questions,
        kor_smos_questions=kor_smos_questions
    )
    
    # Write html into file
    with open("test.html", "w", encoding="utf-8") as f:
        f.write(html)
        print("Done! Generated test.html")
        print(f"  - ENG: {len(eng_mos_questions)} NMOS + {len(eng_smos_questions)} SMOS questions")
        print(f"  - KOR: {len(kor_mos_questions)} NMOS + {len(kor_smos_questions)} SMOS questions")
        print(f"  - Total: {len(eng_mos_questions) + len(eng_smos_questions) + len(kor_mos_questions) + len(kor_smos_questions)} questions")


if __name__ == "__main__":
    main()
