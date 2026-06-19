from vllm import LLM, SamplingParams
from function_load_data import read_argument, load_dataframe, load_concerned_column
from transformers import AutoTokenizer
import argparse
import os
import yaml
import sys
import pdb
from jload import jsave, jload


from huggingface_hub import login
login(token=os.getenv("HUGGINGFACE_TOKEN"))

def load_prompts(prompts_path: str) -> dict:
    """Load prompt templates from a YAML file."""
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    return prompts or {}

def collect_variables(var_args: list[str], text: str | None) -> dict:
    """Collect template variables from --var args and optional --input JSON file."""
    variables: dict = {}

    # if input_file:
    #     with open(input_file, "r", encoding="utf-8") as f:
    #         variables.update(json.load(f))

    for item in var_args or []:
        if "=" not in item:
            print(f"Warning: ignoring malformed --var '{item}' (expected key=value)", file=sys.stderr)
            continue
        key, value = item.split("=", 1)
        variables[key] = text

    return variables


def build_message(prompt_template: dict, variables: dict) -> list[dict]:
    """Build the messages list from a prompt template, substituting variables."""
    message = []
    system = prompt_template.get("system")
    if system:
        message.append({"role": "system", "content": system.format(**variables)})

    user = prompt_template.get("user", "")
    message.append({"role": "user", "content": user.format(**variables)})

    return message

def main():
############ Step 1: Prepare tokenizer and model
# Define your text generation hyperparameters
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=4096)

# Initialize the model (vLLM downloads it automatically from Hugging Face if not cached)
    model_id = "Goedel-LM/Goedel-Prover-V2-8B"
    llm = LLM(model=model_id, seed=42, trust_remote_code=True, max_model_len=8196, tensor_parallel_size=1, pipeline_parallel_size=1, distributed_executor_backend="ray")
    hf_tokenizer_for_chat_template = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

############ Step 2: Prepare data 
    


############ Step 3: Prepare prompt

    
    parser = argparse.ArgumentParser(
        description="Call the Deepseek API with prompts managed in an external YAML file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('-f', '--file_path')
    parser.add_argument('-col', '--column')
    parser.add_argument(
        "--prompts-file",
        default=os.path.join(os.path.dirname(__file__), "..", "prompts_RHIM", "prompts.yaml"),
        help="Path to the YAML prompts file (default: prompts_RHIM/prompts.yaml in the project root)",
    )
    parser.add_argument(
        "--prompt", "-p",
        default="general_chat",
        help="Name of the prompt template to use (default: general_chat)",
    )
    parser.add_argument(
        "--var", "-v",
        dest="variables",
        action="append",
        metavar="KEY=VALUE",
        help="Set a template variable. Use KEY=@file to read the value from a file. Repeatable.",
    )
    
    parser.add_argument(
        "--input", "-i",
        help="JSON file containing template variables (overridden by --var)",
    )

    args = parser.parse_args()
    data = load_dataframe(args.file_path)
    data = data.iloc[:4,:]
    problems = load_concerned_column(data, args.column)
    prompts = load_prompts(args.prompts_file)
    prompt_template = prompts[args.prompt] 
    messages = []

    for _, problem in enumerate(problems):
        variables = collect_variables(args.variables, problem)
        # Build messages
        message = build_message(prompt_template, variables)
        messages.append(hf_tokenizer_for_chat_template.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
)        
    outputs = llm.generate(messages, sampling_params)
    
    responses = []
    for _, (problem, output) in enumerate(zip(problems, outputs)):
        e = dict()
        e["id"] = data.iloc[_,:]['Link_API']
        e["Problem"] = problem
        e["FormalProblem"] = output.outputs[0].text
        responses.append(e)
    jsave(responses, "dataset/formalproblem.json")




if __name__ == "__main__":
    main()
