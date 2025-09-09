import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate
from llama_index.core import Settings
# adapted from https://huggingface.co/Writer/camel-5b-hf

from datetime import datetime
from pydantic import BaseModel, Field, List
from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock, MessageRole


query_wrapper_prompt = PromptTemplate(
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{query_str}\n\n### Response:"
)
llm = HuggingFaceLLM(
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="openai/gpt-oss-20b",
    model_name="openai/gpt-oss-20b",
    device_map="auto",
    tokenizer_kwargs={"max_length": 2048},
)


class nutrition(BaseModel):
    """Representation of energy and nutrition content of meal from image/description."""

    kcal: float = Field(
        description="The total energy content of the meal in kcal"
    )
    cho: float = Field(description="The total carbohydrate content of the meal")
    fat: float = Field(description="The total fat content of the meal")
    protein: float = Field(description="The total protein content of the meal")


class FoodItem(BaseModel):
    """Food item recognized"""
    id: int = Field (description="id of identified food item")
    food_item: str = Field (description="identified food item")
class Recognition(BaseModel):
    """Representation of food item list"""
    food_items: list[FoodItem] = Field(description="A description of all food items provided in image")


sllm = llm.as_structured_llm(nutrition)
knowledge="per 100g cookie, pumpkin there is 68.7 g carbs, 450 kcal, 6.2g protein, and 18.1g fat, per 100g Coffee, espresso there is 1.67g carbs, 9 kcal, 0.12 protein, and 0.18 fat" #ideally you should extract this automatically by matching to the excel provided - usda fdc - but manually providing is also acceptable
prompt=f"You are nutrition assistant - estimating nutrition content: energy, carbohydrate, fat,, and protein - based on the following knowledge - {knowledge}"
message="10g of cookie, pumpkin and 100g of coffee, espresso"
system_msg = ChatMessage(role=MessageRole.SYSTEM, blocks=[
            TextBlock(block_type="text", text=prompt)
        ])
user_msg = ChatMessage(role=MessageRole.USER, blocks=[
            TextBlock(block_type="text", text=message)
        ])


resp=sllm.chat([system_msg, user_msg])
# ChatResponse(message=ChatMessage(role=<MessageRole.ASSISTANT: 'assistant'>, additional_kwargs={}, blocks=[TextBlock(block_type='text', text='{"kcal":54.5,"cho":7.87,"fat":1.81,"protein":0.62}')]), raw=nutrition(kcal=54.5, cho=7.87, fat=1.81, protein=0.62), delta=None, logprobs=None, additional_kwargs={})




import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate
from llama_index.core import Settings
# adapted from https://huggingface.co/Writer/camel-5b-hf

from datetime import datetime
from pydantic import BaseModel, Field, List
from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock, MessageRole
from PIL import Image

query_wrapper_prompt = PromptTemplate(
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{query_str}\n\n### Response:"
)
llm = HuggingFaceLLM(
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="google/gemma-3-27b-it",
    model_name="google/gemma-3-27b-it",
    device_map="auto",
    tokenizer_kwargs={"max_length": 2048},
)


class nutrition(BaseModel):
    """Representation of energy and nutrition content of meal from image/description."""

    kcal: float = Field(
        description="The total energy content of the meal in kcal"
    )
    cho: float = Field(description="The total carbohydrate content of the meal")
    fat: float = Field(description="The total fat content of the meal")
    protein: float = Field(description="The total protein content of the meal")


class FoodItem(BaseModel):
    """Food item recognized"""
    id: int = Field (description="id of identified food item")
    food_item: str = Field (description="identified food item")
class Recognition(BaseModel):
    """Representation of food item list"""
    food_items: list[FoodItem] = Field(description="A description of all food items provided in image")


sllm = llm.as_structured_llm(nutrition)
knowledge="per 100g cookie, pumpkin there is 68.7 g carbs, 450 kcal, 6.2g protein, and 18.1g fat, per 100g Coffee, espresso there is 1.67g carbs, 9 kcal, 0.12 protein, and 0.18 fat" #ideally you should extract this automatically by matching to the excel provided - usda fdc - but manually providing is also acceptable
prompt=f"You are nutrition assistant - estimating totak energy and nutrition content from images: energy, carbohydrate, fat,, and protein - based on the following knowledge - {knowledge}"
message="What is the energy and nutritional content of my meal if you know that the image contains espresso and cookie, pumpkin"
system_msg = ChatMessage(role=MessageRole.SYSTEM, blocks=[
            TextBlock(block_type="text", text=prompt)
        ])

image=Image.open("coffee_espresso.jpg")
user_msg = ChatMessage(role=MessageRole.USER, blocks=[
            TextBlock(block_type="text", text=message),
            ImageBlock(block_type="image",text=image)
        ])


resp=sllm.chat([system_msg, user_msg])
# ChatResponse(message=ChatMessage(role=<MessageRole.ASSISTANT: 'assistant'>, additional_kwargs={}, blocks=[TextBlock(block_type='text', text='{"kcal":294.35,"cho":41.35,"fat":10.09,"protein":3.21}')]), raw=nutrition(kcal=294.35, cho=41.35, fat=10.09, protein=3.21), delta=None, logprobs=None, additional_kwargs={})





import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate
from llama_index.core import Settings
# adapted from https://huggingface.co/Writer/camel-5b-hf

from datetime import datetime
from pydantic import BaseModel, Field, List
from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock, MessageRole
from PIL import Image

query_wrapper_prompt = PromptTemplate(
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{query_str}\n\n### Response:"
)
llm = HuggingFaceLLM(
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="google/gemma-3-27b-it",
    model_name="google/gemma-3-27b-it",
    device_map="auto",
    tokenizer_kwargs={"max_length": 2048},
)


class nutrition(BaseModel):
    """Representation of energy and nutrition content of meal from image/description."""

    kcal: float = Field(
        description="The total energy content of the meal in kcal"
    )
    cho: float = Field(description="The total carbohydrate content of the meal")
    fat: float = Field(description="The total fat content of the meal")
    protein: float = Field(description="The total protein content of the meal")


class FoodItem(BaseModel):
    """Food item recognized"""
    id: int = Field (description="id of identified food item")
    food_item: str = Field (description="identified food item")
class Recognition(BaseModel):
    """Representation of food item list"""
    food_items: list[FoodItem] = Field(description="A description of all food items provided in image")


sllm = llm.as_structured_llm(Recognition)
prompt=f"You are a meal image analyser providing information of the foods you see in an image"
message="Identify the foods in my meal image"
system_msg = ChatMessage(role=MessageRole.SYSTEM, blocks=[
            TextBlock(block_type="text", text=prompt)
        ])

image=Image.open("coffee_espresso.jpg")
user_msg = ChatMessage(role=MessageRole.USER, blocks=[
            TextBlock(block_type="text", text=message),
            ImageBlock(block_type="image",text=image)
        ])


resp=sllm.chat([system_msg, user_msg])
# ChatResponse(message=ChatMessage(role=<MessageRole.ASSISTANT: 'assistant'>, additional_kwargs={}, blocks=[TextBlock(block_type='text', text='{"food_items":[{"id":1,"food_item":"Pasta"},{"id":2,"food_item":"Tomato Sauce"},{"id":3,"food_item":"Meatballs"},{"id":4,"food_item":"Parmesan Cheese"}]}')]), raw=Recognition(food_items=[FoodItem(id=1, food_item='Pasta'), FoodItem(id=2, food_item='Tomato Sauce'), FoodItem(id=3, food_item='Meatballs'), FoodItem(id=4, food_item='Parmesan Cheese')]), delta=None, logprobs=None, additional_kwargs={})