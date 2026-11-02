# Every test related to transformer functionality and token processing
import numpy as np
from data import *

text = load_text(path="text_dataset/input.txt")

print(len(text)) #Should be ca. 1.1m
print(text[:10]) 

tokenizer = CharTokenizer(text)
print(tokenizer.stoi)
print(tokenizer.itos)
print(tokenizer.V)

#print(tokenizer.encode('aaaaäaaaa')) #Throws error because of invalid character ä
print(tokenizer.encode('abcde fgh!'))

#print(tokenizer.decode(np.array([1, 5, 7, 12, 8, 2, -1, 65])))
print(tokenizer.decode(tokenizer.encode('abcde fgh!'))) # Should return original string

print(get_text_batch(data=tokenizer.encode(text), batch_size=128, block_size=20)[0][:3, :])
print(get_text_batch(data=tokenizer.encode(text), batch_size=128, block_size=20)[1][:3, :]) # Rows are shifted over by one
