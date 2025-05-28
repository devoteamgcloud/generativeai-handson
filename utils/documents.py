import bs4
import re
from langchain_community.document_loaders import RecursiveUrlLoader

def bs4_extractor(html: str) -> str:
    soup = bs4.BeautifulSoup(
        html,
        "lxml",
        parse_only=bs4.SoupStrainer(
            class_=("page-content", "content", "main-content")
        ),
    )
    return re.sub(r"\\n\\n+", "\\n\\n", soup.text).strip()

def get_documents():
    loader = RecursiveUrlLoader(
        url="https://www.melexis.com/en/tech-info/quality/sustainability",
        max_depth=3,
        prevent_outside=False,
        extractor=bs4_extractor,
    )
    docs = loader.load()
    return docs
