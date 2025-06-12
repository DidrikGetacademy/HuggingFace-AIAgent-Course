import asyncio

#We can also draw workflows. Let’s use the draw_all_possible_flows function to draw the workflow. This stores the workflow in an HTML file.
from llama_index.utils.workflow import draw_all_possible_flows
#-----------------------#
#Basic Workflow Creation#
#-----------------------#
#We can create a single-step workflow by defining a class that inherits from Workflow and decorating your functions with @step.    
#We will also need to add StartEvent and StopEvent, which are special events that are used to indicate the start and end of the workflow.

from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step
class MyWorkflow(Workflow):
    @step 
    async def my_step(self, ev: StartEvent) -> StopEvent:
        #Do something here
        return StopEvent(result="Hello, world!")
    


async def run_workflow(): 
    w = MyWorkflow(timeout=10, verbose=False)
    result = await w.run()
    print(f"result: {result}")
    draw_all_possible_flows(w, "MultiStepWorkflow(Basic Workflow Creation).html")





#-------------------------#
#Connecting Multiple Steps#
#-------------------------#
#To connect multiple steps, we create custom events that carry data between steps. To do so, we need to add an Event that is passed between the steps and transfers the output of the first step to the second step.
from llama_index.core.workflow import Event

class processingEvent(Event):
    intermediate_result: str
    

class MultiStepWorkflow(Workflow):
    @step 
    async def step_one(self, ev: StartEvent) -> processingEvent: #The type hinting is important here, as it ensures that the workflow is executed correctly. Let’s complicate things a bit more!
        # Process inital data
        return processingEvent(intermediate_result="Step 1 complete")
    
    @step
    async def step_two(self, ev: processingEvent) -> StopEvent: #The type hinting is important here, as it ensures that the workflow is executed correctly. Let’s complicate things a bit more!
        # Use the intermediate result
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)
    
async def run_multistepworkflow():
    w = MultiStepWorkflow(timeout=10, verbose=False)
    result = await w.run()
    print(result) #output: Finished processing: Step 1 complete
    draw_all_possible_flows(w, "MultiStepWorkflow(connecting multiple steps).html")




#-----------------#
#Loops and Branches#
#------------------#
#The type hinting is the most powerful part of workflows because it allows us to create branches, loops, and joins to facilitate more complex workflows.
#Let’s show an example of creating a loop by using the union operator |. In the example below, we see that the LoopEvent is taken as input for the step and can also be returned as output.
from llama_index.core.workflow import Event
import random  
class ProcessingEvent(Event):
    intermediate_result: str


class LoopEvent(Event):
    loop_output: str


class MultiStepWorkflow(Workflow):
    @step 
    async def step_one(self, ev: StartEvent | LoopEvent) -> processingEvent | LoopEvent:
        if random.randint(0, 1) == 0:
            print("Bad thing happend")
            return LoopEvent(loop_output="Bak to step one.")
        else:
            print("Good thing happend")
            return processingEvent(intermediate_result="First step complete.")
        
    @step 
    async def step_two(self, ev: processingEvent) -> StopEvent:
        #Use the intermediate result
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)
    
async def  run_loop_and_branches():
    w = MultiStepWorkflow(verbose=False)
    result = await w.run()
    result
    print(f"Result: {result}")
    draw_all_possible_flows(w, "MultiStepWorkflow(Loops and Branches).html")







#There is one last cool trick that we will cover in the course, which is the ability to add state to the workflow.
#----------------#
#State Management#
#----------------#
#State management is useful when you want to keep track of the state of the workflow, so that every step has access to the same state. We can do this by using the Context type hint on top of a parameter in the step function.
from llama_index.core.workflow import Context, StartEvent, StopEvent

@step
async def query(self, ctx: Context, ev: StartEvent) -> StopEvent:
    #Store the query in the context
    await ctx.set("query", "What is the capital of france?")

    #exsample send question to a function that search for the answer...
    val = await web_search_query(query)

    #retrieve query from the contex
    query = await ctx.get("query")

    return StopEvent(result=val)



if __name__ == "__main__":
   #asyncio.run(run_workflow())
   #asyncio.run(run_multistepworkflow())
   #asyncio.run(run_loop_and_branches())



