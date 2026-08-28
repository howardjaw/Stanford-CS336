import regex as re

"""
Unicode conversion
"""

# char = chr(1057)

# print(char)


"""
Encode - decode
"""

# test_string = "hello! こんにちは!"

# utf8_encoded = test_string.encode("utf-8")

# print(utf8_encoded)

# print(list(utf8_encoded))

# print(utf8_encoded.decode("utf-8"))


"""
Pre-tokenizer
"""

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

corpus = "The capital of France is Paris"

lst=[]

for i in re.finditer(PAT, corpus):
    lst.append(i.group())

print(lst)







