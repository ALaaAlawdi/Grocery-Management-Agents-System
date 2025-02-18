# -*- coding: utf-8 -*-
"""
Grocery Management System

This script automates grocery management by extracting data from receipts,
estimating expiration dates, tracking inventory, and recommending recipes.
"""

import os
from crewai import Agent, Task, Crew
from markdown import markdown
from crewai_tools import WebsiteSearchTool


# ==============================
# Step 1: Environment Setup
# ==============================

def setup_environment():
    """Set up environment variables for API keys."""
    os.environ["OPENAI_API_KEY"] = "[YOUR OPENAI API KEY]"
    os.environ["LLAMA_OCR_API_KEY"] = "[YOUR LLAMA OCR API KEY]"


# ==============================
# Step 2: Load Receipt Data
# ==============================

def load_receipt(file_path):
    """
    Load and parse the receipt markdown file.

    Args:
        file_path (str): Path to the markdown receipt file.

    Returns:
        str: Parsed receipt content.
    """
    with open(file_path, 'r') as f:
        return markdown(f.read())


# ==============================
# Step 3: Define Agents
# ==============================

def create_agents():
    """Create agents for the grocery management system."""
    # Receipt Interpreter Agent
    receipt_interpreter_agent = Agent(
        role="Receipt Markdown Interpreter",
        goal=(
            "Extract items, their counts, and weights with units from a receipt in markdown format. "
            "Provide structured data to support the grocery management system."
        ),
        backstory=(
            "As a key member of the grocery management crew, this agent extracts details such as item names, "
            "quantities, and weights from receipt markdown files. Its role is vital for tracking inventory levels."
        ),
        personality="Diligent, detail-oriented, and efficient.",
        allow_delegation=False,
        verbose=True
    )

    # Expiration Date Estimation Agent
    expiration_date_search_web_tool = WebsiteSearchTool(website='https://www.stilltasty.com/')
    expiration_date_search_agent = Agent(
        role="Expiration Date Estimation Specialist",
        goal=(
            "Estimate the expiration dates of items extracted by the Receipt Interpreter Agent. "
            "Utilize online sources to determine typical shelf life when refrigerated."
        ),
        backstory=(
            "This agent ensures groceries are consumed before expiration by searching for shelf-life estimates online."
        ),
        personality="Meticulous, resourceful, and reliable.",
        allow_delegation=False,
        verbose=True,
        tools=[expiration_date_search_web_tool]
    )

    # Grocery Tracker Agent
    grocery_tracker_agent = Agent(
        role="Grocery Inventory Tracker",
        goal=(
            "Track remaining groceries based on user consumption input. "
            "Update the inventory list and provide expiration dates."
        ),
        backstory=(
            "This agent ensures groceries are accurately tracked, minimizing waste and helping users stay organized."
        ),
        personality="Helpful, detail-oriented, and responsive.",
        allow_delegation=False,
        verbose=True
    )

    # Recipe Recommendation Agent
    recipe_web_tool = WebsiteSearchTool(website='https://www.americastestkitchen.com/recipes')
    recipe_recommendation_agent = Agent(
        role="Grocery Recipe Recommendation Specialist",
        goal=(
            "Recommend recipes using available ingredients. "
            "Suggest restocking recommendations if ingredients are insufficient."
        ),
        backstory=(
            "This agent helps households make the most out of their remaining groceries by finding suitable recipes."
        ),
        personality="Creative, resourceful, and efficient.",
        allow_delegation=False,
        verbose=True,
        tools=[recipe_web_tool],
        human_input=True
    )

    return (
        receipt_interpreter_agent,
        expiration_date_search_agent,
        grocery_tracker_agent,
        recipe_recommendation_agent
    )


# ==============================
# Step 4: Define Tasks
# ==============================

def define_tasks(receipt_markdown, today, agents):
    """
    Define tasks for the grocery management system.

    Args:
        receipt_markdown (str): Parsed receipt content.
        today (str): Today's date in YYYY-MM-DD format.
        agents (tuple): Tuple of agents created in `create_agents`.

    Returns:
        list: List of tasks.
    """
    (
        receipt_interpreter_agent,
        expiration_date_search_agent,
        grocery_tracker_agent,
        recipe_recommendation_agent
    ) = agents

    # Task: Read the Receipt
    read_receipt_task = Task(
        agent=receipt_interpreter_agent,
        description=(
            f"Analyze the receipt markdown file: {receipt_markdown}. "
            "Extract information on items purchased, their counts, weights, and units. "
            f"Today's date is {today}. Ensure all item names are clear and human-readable."
        ),
        expected_output={
            "items": [
                {
                    "item_name": "string - Human-readable name of the item",
                    "count": "integer - Number of units purchased",
                    "unit": "string - Unit of measurement (e.g., kg, lbs, pcs)"
                }
            ],
            "date_of_purchase": "string - Date in YYYY-MM-DD format"
        }
    )

    # Task: Estimate Expiration Dates
    expiration_date_search_task = Task(
        agent=expiration_date_search_agent,
        description=(
            "Using the list of items extracted by the Receipt Interpreter Agent, search online to find the typical shelf life of each item. "
            "Add this information to the purchase date to estimate the expiration date for each item."
        ),
        expected_output={
            "items": [
                {
                    "item_name": "string - Human-readable name of the item",
                    "count": "integer - Number of units purchased",
                    "unit": "string - Unit of measurement (e.g., kg, lbs, pcs)",
                    "expiration_date": "string - Estimated expiration date in YYYY-MM-DD format"
                }
            ]
        },
        context=[read_receipt_task]
    )

    # Task: Track Groceries
    grocery_tracking_task = Task(
        agent=grocery_tracker_agent,
        description=(
            "Using the grocery list with expiration dates, update the inventory based on user input about consumed items. "
            "Subtract consumed quantities and provide a summary of what's left, including expiration dates."
        ),
        expected_output={
            "items": [
                {
                    "item_name": "string - Human-readable name of the item",
                    "count": "integer - Updated number of units remaining",
                    "unit": "string - Unit of measurement (e.g., kg, lbs, pcs)",
                    "expiration_date": "string - Estimated expiration date in YYYY-MM-DD format"
                }
            ]
        },
        context=[expiration_date_search_task],
        human_input=True,
        output_file="../data/grocery_management_agents_system/output/grocery_tracker.json"
    )

    # Task: Recommend Recipes
    recipe_recommendation_task = Task(
        agent=recipe_recommendation_agent,
        description=(
            "Using the updated grocery list, search online for recipes that utilize available ingredients. "
            "If no suitable recipe can be found, provide restocking recommendations."
        ),
        expected_output={
            "recipes": [
                {
                    "recipe_name": "string - Name of the recipe",
                    "ingredients": [
                        {
                            "item_name": "string - Ingredient name",
                            "quantity": "string - Quantity required",
                            "unit": "string - Measurement unit (e.g., kg, pcs, tbsp)"
                        }
                    ],
                    "steps": ["string - Step-by-step instructions for the recipe"],
                    "source": "string - Website URL for the recipe"
                }
            ],
            "restock_recommendations": [
                {
                    "item_name": "string - Name of the item to restock",
                    "quantity_needed": "integer - Suggested quantity to purchase",
                    "unit": "string - Measurement unit (e.g., kg, pcs)"
                }
            ]
        },
        context=[grocery_tracking_task],
        output_file="../data/grocery_management_agents_system/output/recipe_recommendation.json"
    )

    return [
        read_receipt_task,
        expiration_date_search_task,
        grocery_tracking_task,
        recipe_recommendation_task
    ]


# ==============================
# Step 5: Run the Crew
# ==============================

def run_crew(tasks, agents):
    """
    Run the crew to execute the defined tasks.

    Args:
        tasks (list): List of tasks.
        agents (tuple): Tuple of agents.

    Returns:
        dict: Results of the crew execution.
    """
    crew = Crew(
        agents=list(agents),
        tasks=tasks,
        verbose=True
    )
    return crew.kickoff()


# ==============================
# Main Execution
# ==============================

if __name__ == "__main__":
    # Step 1: Set up environment
    setup_environment()

    # Step 2: Load receipt data
    receipt_file_path = "../data/grocery_management_agents_system/extracted/grocery_receipt.md"
    receipt_markdown = load_receipt(receipt_file_path)
    today = "2024-11-16"

    # Step 3: Create agents
    agents = create_agents()

    # Step 4: Define tasks
    tasks = define_tasks(receipt_markdown, today, agents)

    # Step 5: Run the crew
    result = run_crew(tasks, agents)
    print("Crew execution completed. Results saved to output files.")