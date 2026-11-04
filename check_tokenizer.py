# Manual checks of the tokenizer and batching. Prints values to compare by eye.
import numpy as np

from data import CharTokenizer, get_text_batch, load_text

text = load_text(path="text_dataset/input.txt")
rng = np.random.default_rng(seed=128)

print(len(text))  # Should be about 1.1 million characters
print(text[:10])

tokenizer = CharTokenizer(text)
print(tokenizer.stoi)
print(tokenizer.itos)
print(tokenizer.V)

# print(tokenizer.encode('aaaaäaaaa'))  # Throws error because of invalid character ä
print(tokenizer.encode("abcde fgh!"))

# print(tokenizer.decode(np.array([1, 5, 7, 12, 8, 2, -1, 65])))  # Throws error for invalid ids
print(tokenizer.decode(tokenizer.encode("abcde fgh!")))  # Should return original string

x_batch, y_batch = get_text_batch(
    data=tokenizer.encode(text), batch_size=128, block_size=20, rng=rng
)
print(x_batch[:3, :])
print(y_batch[:3, :])  # Rows are shifted over by one
