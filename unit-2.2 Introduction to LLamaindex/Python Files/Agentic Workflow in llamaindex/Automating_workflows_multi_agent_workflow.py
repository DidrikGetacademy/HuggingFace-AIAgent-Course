#Automating workflows with Multi-Agent Workflows
    #Instead of manual workflow creation
    #we can use the AgentWorkflow class to create a multi-agent workflow. 
    #The AgentWorkflow uses Workflow Agents to allow you to create a system of one or more agents that can collaborate and hand off tasks to each other based on their specialized capabilities.
    #This enables building complex agent systems where different agents handle different aspects of a task. Instead of importing classes from llama_index.core.agent, we will import the agent classes from llama_index.core.agent.workflow. 
    #One agent must be designated as the root agent in the AgentWorkflow constructor. When a user message comes in, it is first routed to the root agent.

    #Each agent can then
        # Handle the request directly using their tools 

        # handoff to another agent better suited for the task

        # return a response to the user

import asyncio
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.utils.workflow import draw_all_possible_flows



async def run_workflow():
        #Define some tools
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

    # we can pass functions directly without FunctionTool -- the fn/docstring are parsed for the name/description

    multiply_agent = ReActAgent(
        name="multiply_agent",
        description="Is able to multiply two integers",
        system_prompt="A helpful assistant that can use a tool to multiply numbers",
        tools=[multiply],
        llm=llm,
    )

    addition_agent = ReActAgent(
        name="add_agent",
        description="Is able to add two integers",
        system_prompt="A helpful assistant that can use a tool to add numbers.",
        tools=[add],
        llm=llm,
    )

    workflow = AgentWorkflow(
        agents=[multiply_agent,addition_agent],
        root_agent="multiply_agent",
    )
    draw_all_possible_flows(workflow, "workflowworkflow.html")
    response = await workflow.run(user_msg="Can you ask the add_agent to add 999 and 1? also explain what you did too achieve the result. explain chain of thought",verbose=True)
    print(f"response: {response}")




# Agent tools can also modify the workflow state we mentioned earlier. Before starting the workflow, we can provide an initial state dict that will be available to all agents. 
# The state is stored in the state key of the workflow context. It will be injected into the state_prompt which augments each new user message.
# Let’s inject a counter to count function calls by modifying the previous example:


async def run_workflow_with_state():
    from llama_index.core.workflow import Context

    async  def add(ctx: Context, a: int, b: int) -> int:
        """Add two numbers"""
        #Update our count
        current_state = await ctx.get("state")
        current_state["num_fn_calls"] += 1
        await ctx.set("state", current_state)

        return a + b

    async def multiply(ctx: Context, a: int, b: int) -> int:
        """Multiply two numbers."""
        #Update our count

        current_state = await ctx.get("state")
        current_state["num_fn_calls"] += 1
        await ctx.set("state", current_state)
        return a * b

    llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")


    multiply_agent = ReActAgent(
        name="multiply_agent",
        description="Is able to multiply two integers",
        system_prompt="A helpful assistant that can use a tool to multiply numbers",
        tools=[multiply],
        llm=llm,
    )

    addition_agent = ReActAgent(
        name="add_agent",
        description="Is able to add two integers",
        system_prompt="A helpful assistant that can use a tool to add numbers.",
        tools=[add],
        llm=llm,
    )

    workflow = AgentWorkflow(
        agents=[multiply_agent,addition_agent],
        root_agent="multiply_agent",
        initial_state={"num_fn_calls": 0},
        state_prompt="Current state: {state}. User message: {msg} and you detailed chain of thought explaining every step to achieve the task"
    )
    draw_all_possible_flows(workflow, "workflowworkflow.html")

    # run the workflow with context
    ctx = Context(workflow)
    response = await workflow.run(user_msg=" Can you add 5 and 3?  then can you multiply 5 and 2",ctx=ctx, verbose=True)
    print(f"response: {response}")

    # pull out and inspect the state
    state = await ctx.get("state")
    print(state["num_fn_calls"]) #output will be 2. because it uses the tools from each agent. 


if __name__ == "__main__":
    #asyncio.run(run_workflow())
    asyncio.run(run_workflow_with_state())