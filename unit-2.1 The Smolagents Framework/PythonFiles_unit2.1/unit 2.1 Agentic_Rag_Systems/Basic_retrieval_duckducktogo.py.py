from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel,TransformersModel


search_tool = DuckDuckGoSearchTool()

model = HfApiModel()

agent = CodeAgent(
        model=model,
        tools=[search_tool]
    )
response = agent.run("Search for luxury superhero-Themed party ideas, including decorations, entertainment, and catering.")
print(response)







