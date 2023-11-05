import transformers
import evaluate
import torch
import json
import random
from tqdm import tqdm
from datasets import load_dataset
import argparse

def read_file(fn):
	with open(fn) as f:
		data = f.read()
	return data

def write_pretty_json(file_path, data):
    with open(file_path, "w") as write_file:
        json.dump(data, write_file, indent=4)
    print(f"wrote {file_path}")

model_path="models/OpenHermes-2-Mistral-7B"
input_file="kc_reformat.txt"

file_content=read_file(input_file)
chapters=file_content.split("\n\n")
paragraphs=file_content.split("\n")
passage_minlen=300
passage_maxlen=2000
outputfn=input_file.split(".")[0]+"_interview.json"

passages=[]
for chap in chapters:
	passage=""
	for par in chap.split("\n"):
		if(len(passage)<passage_minlen) or not passage[-1]=="." and len(passage)<passage_maxlen:
			passage+="\n" + par
		else:
			passages.append(passage.strip().replace("\n", " "))
			passage=par

prompt_template="""<|im_start|>system
You are an expert interviewer who interviews an autobiography of a famous chef. You formulate questions based on quotes from the autobiography. Below is one such quote. Formulate a question that the quote would be the perfect answer to. The question should be short and directed at the author of the autobiography like in an interview. The question is short. Remember, make the question as short as possible. Do not give away the answer in your question. Also: If possible, ask for motvations, feelings, and perceptions rather than events or facts.

Here is some context that might help you formulate the question regarding the quote:
{ctx}
<|im_end|>
<|im_start|>user
Quote:
{par}<|im_end|>
<|im_start|>assistant
Question:"""

prompts=[]
for i,p in enumerate(passages):
	if i==0:
		continue
	prompt=prompt_template.format(par=passages[i], ctx=passages[i-1]) 
	prompts.append(prompt)

prompts_generator=(p for p in prompts)	# pipeline needs a generator, not a list

print(f"{len(chapters)} chapters")
print(f"{len(paragraphs)} paragraphs")
print(f"{len(passages)} passages")

pipeline = transformers.pipeline(
		"text-generation",
		model=model_path,
		torch_dtype=torch.bfloat16,
		device_map="auto",
	)

pipeline.tokenizer.add_special_tokens({"pad_token":"<pad>"})
pipeline.model.resize_token_embeddings(len(pipeline.tokenizer))
pipeline.model.config.pad_token_id = pipeline.tokenizer.pad_token_id

gen_config = {
    "temperature": 0.7,
    "top_p": 0.1,
    "repetition_penalty": 1.18,
    "top_k": 40,
	"do_sample": True,
	"num_return_sequences": 1,
	"eos_token_id": pipeline.tokenizer.eos_token_id,
	"max_new_tokens": 50,		
}

results={
	"model": model_path,
	"input_file": input_file,
	"gen_config": gen_config,
	"passage_minlen": passage_minlen,
	"passage_maxlen": passage_maxlen,
	"num_passages": len(passages),
	"template": prompt_template,
	"interview": []
}

for i, out in enumerate(tqdm(pipeline(prompts_generator, batch_size=2, **gen_config),total=len(prompts))):
	question=out[0]["generated_text"][len(prompts[i]):].strip()
	answer=passages[i+1]

	results["interview"].append({"question": question, "answer": answer})

	write_pretty_json(outputfn,results)
