
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
# Import two classes from llama_index that help you connect to and use an MCP server:
# - BasicMCPClient: A client that connects to a running MCP server via a URL.
# - McpToolSpec: Wraps the MCP client as a tool that can be used by an agent.
import asyncio
#LlamaIndex also allows using MCP tools through a ToolSpec on the LlamaHub.
    # You can simply run an MCP server and start using it through the following implementation.

async def main():

    mcp_client = BasicMCPClient("http://127.0.0.1:8000/sse")# We consider there is a mcp server running on 127.0.0.1:8000, or you can use the mcp client to connect to your own mcp server.

    mcp_tool = McpToolSpec(client=mcp_client)
    # Wrap the MCP client into a "tool specification".
    # This makes the MCP client compatible with the LlamaHub framework, so agents can use it as a tool to perform tasks.


    agent = await get_agent(mcp_tool)  #get the agent
    # Call an async function `get_agent` (not shown here) that returns an agent configured with the MCP tool.
    # 'await' means "wait here until this async operation finishes" without blocking the whole program.
    # The agent can use the MCP tool to interact with the MCP server.


    agent_context = Context(agent) # create the agent context
    # Create a Context object for the agent.
    # The context typically stores information/state related to the agent’s current session or environment.
    # This helps the agent maintain continuity in conversations or tasks.

asyncio.run(main())