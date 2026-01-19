#!/usr/bin/env python3
"""Generate ENG-only NMOS & SMOS evaluation form."""

from jinja2 import FileSystemLoader, Environment
from utils import QuestionGenerator_MOS, QuestionGenerator_SMOS


def main():
    """Main function."""
    loader = FileSystemLoader(searchpath="./templates")
    env = Environment(loader=loader)
    template = env.get_template("test_single.html.jinja2")
    
    # Load ENG MOS questions
    eng_mos_questions = QuestionGenerator_MOS("filelist/NMOS_eng_10.csv").questions
    
    # Load ENG SMOS questions
    eng_smos_questions = QuestionGenerator_SMOS("filelist/SMOS_eng_10.csv").questions

    html = template.render(
        page_title="NMOS & SMOS Experiment Form (English)",
        language="ENG",
        language_name="English",
        language_name_kr="영어",
        form_url="https://script.google.com/macros/s/AKfycbz9n8g2pCje9P3VdorsWhPLAVSk-hiZ17Ip8QzyAHyPCU7Wn2Y2HTApJiFIjOrqbpw6/exec",
        mos_questions=eng_mos_questions,
        smos_questions=eng_smos_questions
    )
    
    # Write html into file
    with open("test_eng.html", "w", encoding="utf-8") as f:
        f.write(html)
        print("Done! Generated test_eng.html")
        print(f"  - ENG: {len(eng_mos_questions)} NMOS + {len(eng_smos_questions)} SMOS questions")
        print(f"  - Total: {len(eng_mos_questions) + len(eng_smos_questions)} questions")


if __name__ == "__main__":
    main()
