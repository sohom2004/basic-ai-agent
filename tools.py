from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

def save_to_text(content: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_content = f"---Research Output---\nTimestamp: {timestamp}\n\n{content}\n\n"
    with open(filename, 'a', encoding="utf-8") as file:
        file.write(formatted_content)
    return f"Data saved to {filename}"

save_tool = Tool(
    name="save_to_text",
    func=save_to_text,
    description="Saves the research output to a text file with a timestamp."
)

search = DuckDuckGoSearchRun()

search_tool = Tool(
    name="search",
    func=search.run,
    description="Useful for searching the web for information not found in the knowledge base."
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)