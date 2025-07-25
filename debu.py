from transformers import AutoTokenizer

# Load tokenizer (replace with the correct model name)
tokenizer = AutoTokenizer.from_pretrained("ZhangShenao/Llama-3.2-3B")

# Example token id or list of ids
token_id = 151643             # Single token ID
#token_ids = [50256, 318, 257]  # Multiple token IDs

# Decode
decoded_single = tokenizer.decode([token_id])
#decoded_multiple = tokenizer.decode(token_ids)

print("Single token:", decoded_single, 'zz')
print(tokenizer.encode(" "), tokenizer.encode(""))
#print("Multiple tokens:", decoded_multiple)

