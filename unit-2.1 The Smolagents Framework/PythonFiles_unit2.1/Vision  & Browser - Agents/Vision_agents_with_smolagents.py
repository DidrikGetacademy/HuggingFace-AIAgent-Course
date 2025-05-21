from PIL import Image
import requests
from io import BytesIO
from smolagents import CodeAgent, OpenAIServerModel,TransformersModel,tool, ActionStep,DuckDuckGoSearchTool,WebSearchTool,VisitWebpageTool,HfApiModel,ToolCallingAgent
import torch
from dotenv import load_dotenv
import os
import time
from time import sleep
os.environ["TRANSFORMERS_NO_FLASH_ATTENTION"] = "1"
import os
print("FlashAttention disabled:", os.getenv("TRANSFORMERS_NO_FLASH_ATTENTION"))



def Alfred_catching_joker_image_reading():
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    image_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/e/e8/The_Joker_at_Wax_Museum_Plus.jpg",
        "https://upload.wikimedia.org/wikipedia/en/9/98/Joker_%28DC_Comics_character%29.jpg"
    ]

    images = []

    for url in image_urls:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36" 
        }
        response = requests.get(url, headers=headers)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        images.append(image)

    Model_openai = OpenAIServerModel(model_id="gpt-4o")

    model = TransformersModel(
        model_id=Model_openai,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )



    agent = CodeAgent(
        tools=[],
        model=model,
        max_steps=20,
        verbosity_level=4,
    )

    response = agent.run(
        """
        Describe the costume and makeup that the comic character in these photos is wearing.
        Identify if the character is The Joker or Wonder Woman.
        """,
        images=images,
    )




def images_dynamic_Retrieval():
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.by import By
    import chromedriver_autoinstaller
    chromedriver_autoinstaller.install() 
    from selenium import webdriver
    import helium
    driver = webdriver.Chrome()

    @tool #This tools is passed to agent as step_callback, this is triggered at end of each step during the agents execution, this allows the agent to dynamically captues and store screenshots thoughtout its process 
    def save_screenshot(step_log: ActionStep, agent: CodeAgent):
        sleep(1.0) #a small delay to let website load before taking screenshot
        driver = helium.get_driver()
        current_step = step_log.step_number
        if driver is not None:
            for step_logs in agent.logs:
                if isinstance(step_log, ActionStep) and step_log.step_number <= current_step -2:
                    step_logs.observations_images = None
            png_bytes = driver.get_screenshot_as_png()
            image = Image.open(BytesIO(png_bytes))
            print(f"Captured a browser screenshot: {image.size} pixels")
            step_log.observations_images = [image.copy()] #creates a copy to ensure it persists, important!

        url_info = f"current url: {driver.current_url}"
        
        #update the observation with current url.
        step_log.observations = url_info if step_log.observations is None else step_log.observations + "\n" + url_info


    @tool 
    def search_item_ctrl_f(text:str, nth_result: int=1) -> str:
        """
        Searches for text on the current page via Ctrl + F and jumps to the nth occurrence.
        Args:
            text: The text to search for
            nth_result: Which occurrence to jump to (default: 1)
        """
        elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{text})]")
        if nth_result > len(elements):
            raise Exception(f"Match n°{nth_result} not found (only {len(elements)} matches found)")
        result = f"Found {len(elements)} matches for '{text}'"
        elem = elements[nth_result - 1]
        driver.execute_script("arguments[0].scrollIntoView(True)", elem)
        result += f"Focused on element {nth_result} of {len(elements)}"
        return result

    @tool 
    def go_back() -> None:
        """Goes back to previous page"""
        driver.back()

    @tool
    def close_popups() -> str:
        """
        Closes any visible modal or pop-up on the page. Use this to dismiss pop-up windows! This does not work on cookie consent banners.
        """
        webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform
    import yaml
    with open(r"C:\Users\didri\Desktop\Programmering\AiCognetics Nettside\AI-Agent\Huggingface Agent Course\unit-2.1 The Smolagents Framework\PythonFiles_unit2.1\Vision  & Browser - Agents\VLM_prompts.yaml", "r", "utf-8") as file:
      prompt =  yaml.load(file)

    model = OpenAIServerModel(model_id="gpt-4o")
    agent = CodeAgent(
        model=model,
        tools=[DuckDuckGoSearchTool(),go_back, close_popups, search_item_ctrl_f],
        additional_authorized_imports="helium",
        step_callbacks =[save_screenshot],
        max_steps=20,
        verbosity_level=4,
        prompt_templates=prompt
    )

    helium_instructions = """
    Use your web_search tool when you want to get Google search results.
    Then you can use helium to access websites. Don't use helium for Google search, only for navigating websites!
    Don't bother about the helium driver, it's already managed.
    We've already ran "from helium import *"
    Then you can go to pages!
    Code:
    ```py
    go_to('github.com/trending')
    ```<end_code>

    You can directly click clickable elements by inputting the text that appears on them.
    Code:
    ```py
    click("Top products")
    ```<end_code>

    If it's a link:
    Code:
    ```py
    click(Link("Top products"))
    ```<end_code>

    If you try to interact with an element and it's not found, you'll get a LookupError.
    In general stop your action after each button click to see what happens on your screenshot.
    Never try to login in a page.

    To scroll up or down, use scroll_down or scroll_up with as an argument the number of pixels to scroll from.
    Code:
    ```py
    scroll_down(num_pixels=1200) # This will scroll one viewport down
    ```<end_code>

    When you have pop-ups with a cross icon to close, don't try to click the close icon by finding its element or targeting an 'X' element (this most often fails).
    Just use your built-in tool `close_popups` to close them:
    Code:
    ```py
    close_popups()
    ```<end_code>

    You can use .exists() to check for the existence of an element. For example:
    Code:
    ```py
    if Text('Accept cookies?').exists():
        click('I accept')
    ```<end_code>

    Proceed in several steps rather than trying to solve the task in one shot.
    And at the end, only when you have your answer, return your final answer.
    Code:
    ```py
    final_answer("YOUR_ANSWER_HERE")
    ```<end_code>

    If pages seem stuck on loading, you might have to wait, for instance `import time` and run `time.sleep(5.0)`. But don't overuse this!
    To list elements on page, DO NOT try code-based element searches like 'contributors = find_all(S("ol > li"))': just look at the latest screenshot you have and read it visually, or use your tool search_item_ctrl_f.
    Of course, you can act on buttons like a user would do when navigating.
    After each code blob you write, you will be automatically provided with an updated screenshot of the browser and the current browser url.
    But beware that the screenshot will only be taken at the end of the whole action, it won't see intermediate states.
    Don't kill the browser.
    When you have modals or cookie banners on screen, you should get rid of them before you can click anything else.
    """

   
    agent.run("""
            I am Alfred, the butler of Wayne Manor, responsible for verifying the identity of guests at party. A superhero has arrived at the entrance claiming to be Wonder Woman, but I need to confirm if she is who she says she is.

            Please search for images of Wonder Woman and generate a detailed visual description based on those images. Additionally, navigate to Wikipedia to gather key details about her appearance. With this information, I can determine whether to grant her access to the event.
            """ + helium_instructions)

if __name__ == "__main__":
  images_dynamic_Retrieval()