#!/usr/bin/env python3
"""Generate forms for human evaluation."""

from jinja2 import FileSystemLoader, Environment
from utils import QuestionGenerator_MOS as QuestionGenerator


def main():
    """Main function."""
    loader = FileSystemLoader(searchpath="./templates")
    env = Environment(loader=loader)
    template = env.get_template("nmos.html.jinja2")
    


    html = template.render(
        page_title="NMOS Experiment Form",
        form_url="https://script.google.com/macros/s/AKfycbz88pG3Usw7jdtNWh9pUWnfPR855SESSYZFGyaYbPv006CSZplhdrc2v2wwhjakF8mkjg/exec",  # Replace with your Google Apps Script Web App URL
        questions = QuestionGenerator("filelist/NMOS.csv").questions
    )
    # write html into file
    with open("nmos.html", "w", encoding="utf-8") as f:
        f.write(html)
        print("Done! Generated nmos.html")


if __name__ == "__main__":
    main()