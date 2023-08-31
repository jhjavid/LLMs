from key import Openai_key
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
import os

os.environ['OPENAI_API_KEY'] = Openai_key

llm = OpenAI(temperature=0.7)
prompt = PromptTemplate(
    input_variables = ['movie'],
    template = "who was the actor in {movie}.Tell me the just the name."
)
name_chain = LLMChain(llm=llm,prompt=prompt)
prompt2 = PromptTemplate(
    input_variables = ['actor'],
    template = "what is {actor} actors age.Tell me the just the age."
)
age_chain = LLMChain(llm=llm,prompt=prompt2)
#print('Your Prompt: ',prompt.format(part_of_world='Indian'))

chain = SimpleSequentialChain(chains = [name_chain,age_chain])

print(chain.run('The Pursuit of Happyness'))