from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import  PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

from tools import search_tool, wikipedia_tool, save_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",)
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            """
            You are a helpful research assistant. Your task is to provide a comprehensive research paper.
            Answer the user query and use the necessary tools to gather information.
            Wrap the output in this format and provide no other text\n{format_instructions}
            Use the save_to_text tool to save the research output.
            After you complete the research and structure the response, you **must call the `save_to_text` tool** to save your output. 

            """
            ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
        
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wikipedia_tool, save_tool]

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("Enter your research query: ")
raw_response = agent_executor.invoke({"query": query})

try:
    structured_response = parser.parse(raw_response.get("output"))
except Exception as e:
    print("Error parsing response", e, "Raw response:", raw_response)
print(structured_response)