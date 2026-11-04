# Interactive text generation from a trained GPT checkpoint.
# Type a prompt to continue it, press enter for a fresh start or type STOP to quit.
import numpy as np

from checkpoints import load_gpt_weights


rng = np.random.default_rng(seed=1234)
model, tok = load_gpt_weights("gpt_checkpoints/gpt_shakespeare.npz")

prompt = ""

while True:
    prompt = input("Input your Prompt: ")
    if prompt == "STOP":
        break
    # An empty prompt starts from a newline so the model has at least one token
    if not prompt:
        prompt = "\n"
    # The tokenizer only knows characters that appear in the training text
    try:
        ids = tok.encode(prompt).reshape(1, -1)
    except ValueError as error:
        print(error)
        continue
    out = model.generate(ids, max_new_tokens=500, rng=rng, temperature=0.8)
    text = tok.decode(out[0])
    print(text)
