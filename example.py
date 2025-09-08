import torch
from llama_index.llms.huggingface import HuggingFaceLLM
# setup prompts - specific to StableLM
from llama_index.core import PromptTemplate
from llama_index.core import Settings
# adapted from https://huggingface.co/Writer/camel-5b-hf

from datetime import datetime
from pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock, MessageRole


query_wrapper_prompt = PromptTemplate(
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{query_str}\n\n### Response:"
)
llm = HuggingFaceLLM(
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.25, "do_sample": False},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="openai/gpt-oss-20b",
    model_name="openai/gpt-oss-20b",
    device_map="auto",
    tokenizer_kwargs={"max_length": 2048},
    # uncomment this if using CUDA to reduce memory usage
    # model_kwargs={"torch_dtype": torch.float16}
)


class nutrition(BaseModel):
    """A representation of information from an invoice."""

    kcal: float = Field(
        description="The total energy content of the meal in kcal"
    )
    cho: float = Field(description="The total carbohydrate content of the meal")
    fat: float = Field(description="The total fat content of the meal")
    protein: float = Field(description="The total protein content of the meal")

    


sllm = llm.as_structured_llm(nutrition)

prompt="You are nutrition assistant - estimating nutrition content: energy, carbohydrate, fat,, and protein - based on the following knowledge - per 100g cookie, pumpkin there is 68.7 g carbs, 450 kcal, 6.2g protein, and 18.1g fat, per 100g Coffee, espresso there is 1.67g carbs, 9 kcal, 0.12 protein, and 0.18 fat"
message="10g of cookie, pumpkin and 100g of coffee, espresso"
system_msg = ChatMessage(role=MessageRole.SYSTEM, blocks=[
            TextBlock(block_type="text", text=prompt)
        ])
user_msg = ChatMessage(role=MessageRole.USER, blocks=[
            TextBlock(block_type="text", text=message)
        ])


resp=sllm.chat([system_msg, user_msg])

# >>> resp
# ChatResponse(message=ChatMessage(role=<MessageRole.ASSISTANT: 'assistant'>, additional_kwargs={}, blocks=[TextBlock(block_type='text', text='{"kcal":54,"cho":7,"fat":2,"protein":1}')]), raw=nutrition(kcal=54, cho=7, fat=2, protein=1), delta=None, logprobs=None, additional_kwargs={})