import sys
from gpt4all import GPT4All as gp
from rich import print

model_name = "gpt4all-13b-snoozy-q4_0.gguf"

# Define the valid arguments and their defaults
valid_args = {
    '--max_tokens': 5000,
    '--temp': 0.15,
    '--top_k': 40,
    '--top_p': 0.4,
    '--min_p': 0.0,
    '--repeat_penalty': 1.18,
    '--repeat_last_n': 64,
    '--n_batch': 8,
    '--n_predict': None,
    '--streaming': True,
    '--callback': None,
    '--gpu': False
}

# Initialize a dictionary to store argument values
args_dict = {arg: None for arg in valid_args}

# Loop through the command-line arguments and store the values
for i, arg in enumerate(sys.argv):
    if arg in valid_args:
        if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--'):
            args_dict[arg] = sys.argv[i + 1]
        else:
            args_dict[arg] = None  # For flags that do not have a value

# Convert the arguments and their values to the correct types
model_args = {}
for arg, value in args_dict.items():
    key = arg.lstrip('--')
    if value is not None:
        if isinstance(valid_args[arg], bool):
            model_args[key] = value.lower() == 'true'
        else:
            model_args[key] = type(valid_args[arg])(value)
    else:
        model_args[key] = valid_args[arg]

# Remove 'callback' and 'gpu' key from model_args if it is not a callable
if model_args.get('callback') is not None:
    model_args['callback'] = None
if model_args.get('gpu') is True:
    gpu = gp.list_gpus()
    if gpu:
        model_device = gpu[0]
        print("[bright_yellow]" + model_device)
    else:
        model_device = None
        print("[bright_yellow]No GPU found[/bright_yellow]")
else:
    model_device = None
    print("[bright_yellow]CPU ONLY[/bright_yellow]")

print("[magenta]Model Arguments:[/magenta]")
print(model_args)
print("[red]CTRL+C to Exit[/red]")

def main(model_args):
    model = gp(model_name, device=model_device)

    with model.chat_session():
        while True:
            try:
                print("[cyan]You (Press CTRL+D to submit):[/cyan]")
                prompt_lines = []
                while True:
                    try:
                        line = input()
                    except EOFError:
                        break
                    prompt_lines.append(line)

                prompt = '\n'.join(prompt_lines)

                generate_args = {
                    'prompt': prompt,
                    'max_tokens': model_args.get('max_tokens', 5000),
                    'temp': model_args.get('temp', 0.15),
                    'top_k': model_args.get('top_k', 40),
                    'top_p': model_args.get('top_p', 0.4),
                    'min_p': model_args.get('min_p', 0.0),
                    'repeat_penalty': model_args.get('repeat_penalty', 1.18),
                    'repeat_last_n': model_args.get('repeat_last_n', 64),
                    'n_batch': model_args.get('n_batch', 8),
                    'n_predict': model_args.get('n_predict', None),
                    'streaming': model_args.get('streaming', True),
                }

                # Only include callback if it's not None and is callable
                if model_args.get('callback') is not None and callable(model_args['callback']):
                    generate_args['callback'] = model_args['callback']

                tokens = []

                for token in model.generate(**generate_args):
                    tokens.append(token)
                response = ''.join(tokens)
                print("[green]" + "    GPT-4All:")
                print(response)  # Indent response
            except KeyboardInterrupt:
                print("[red]\nExiting the chat session.[/red]")
                model.close()
                break

if __name__ == "__main__":
    main(model_args)

"""
generate(prompt, *, max_tokens=200, temp=0.7, top_k=40, top_p=0.4, min_p=0.0, repeat_penalty=1.18, repeat_last_n=64, n_batch=8, n_predict=None, streaming=False, callback=empty_response_callback)
Parameters:

prompt (str) – The prompt for the model the complete.
max_tokens (int, default: 200 ) – The maximum number of tokens to generate.
temp (float, default: 0.7 ) – The model temperature. Larger values increase creativity but decrease factuality.
top_k (int, default: 40 ) – Randomly sample from the top_k most likely tokens at each generation step. Set this to 1 for greedy decoding.
top_p (float, default: 0.4 ) – Randomly sample at each generation step from the top most likely tokens whose probabilities add up to top_p.
min_p (float, default: 0.0 ) – Randomly sample at each generation step from the top most likely tokens whose probabilities are at least min_p.
repeat_penalty (float, default: 1.18 ) – Penalize the model for repetition. Higher values result in less repetition.
repeat_last_n (int, default: 64 ) – How far in the models generation history to apply the repeat penalty.
n_batch (int, default: 8 ) – Number of prompt tokens processed in parallel. Larger values decrease latency but increase resource requirements.
n_predict (int | None, default: None ) – Equivalent to max_tokens, exists for backwards compatibility.
streaming (bool, default: False ) – If True, this method will instead return a generator that yields tokens as the model generates them.
callback (ResponseCallbackType, default: empty_response_callback ) – A function with arguments token_id:int and response:str, which receives the tokens from the model as they are generated and stops the generation by returning False.
Returns:

Any – Either the entire completion or a generator that yields the completion token by token.
"""
