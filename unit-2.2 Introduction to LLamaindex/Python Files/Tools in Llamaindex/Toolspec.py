#####ToolSpecs - set of tools created by the community#####


from llama_index.tools.google import GmailToolSpec

# Think of ToolSpecs as collections of tools that work together harmoniously - like a well-organized professional toolkit.
# Just as a mechanic’s toolkit contains complementary tools that work together for vehicle repairs, a ToolSpec combines related tools for specific purposes.
# For example, an accounting agent’s ToolSpec might elegantly integrate spreadsheet capabilities, email functionality, and calculation tools to handle financial tasks with precision and efficiency.

tool_spec = GmailToolSpec()
tool_spec_list = tool_spec.to_tool_list()

#more detailed view of the tools, by taking a look at the metadata:
for tool in tool_spec_list:
    print(f"Name: {tool.metadata.name}\nDescription: {tool.metadata.description}\n")
