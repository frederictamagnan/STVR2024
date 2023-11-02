import torch
torch.cuda.is_available()

from transformers import RobertaForMaskedLM
from transformers import PreTrainedTokenizerFast


tokenizerr = PreTrainedTokenizerFast(tokenizer_file="./data/models/durr")

tokenizerr.add_special_tokens({'mask_token': '<mask>'})
tokenizerr.add_special_tokens({'pad_token': '<pad>'})
print("mask",tokenizerr.mask_token)



model = RobertaForMaskedLM.from_pretrained('./data/models/femto')
sentence = "/Web/reservation.phpchangeselect#BeginPeriod /Web/reservation.phpclickoption#BeginPeriod||>||:nth-child(5) /Web/reservation.phpchangeselect#EndPeriod /Web/reservation.phpclickoption#EndPeriod||>||:nth-child(10)"
input_ids = tokenizerr.encode(sentence, add_special_tokens=True)
# with torch.no_grad():
#     outputs = model(torch.tensor(input_ids).unsqueeze(0))
#     last_hidden_states = outputs[0]

# embeddings = []
# for row in last_hidden_states[0]:
#     embeddings.append(row)

# # print(embeddings)
# from transformers import pipeline

# fill_mask = pipeline(
#     "fill-mask",
#     model="./data/models/femto",
#     tokenizer=tokenizerr
# )
# Tokenize the beginning of the sequence
input_ids = tokenizerr.encode(sentence, add_special_tokens=False, return_tensors="pt")

# # Define the maximum length of the completed sequence
# max_length = len(input_ids) + 5

# # Loop to generate the rest of the sequence
# for i in range(max_length):
#     # Convert the input_ids tensor to a list for editing
#     input_ids_list = input_ids.tolist()[0]
    
#     # Add a mask token to the end of the input_ids list
#     input_ids_list.append(tokenizerr.mask_token_id)
    
#     # Convert the input_ids list back to a tensor
#     input_ids = torch.tensor([input_ids_list])
    
#     # Generate predictions for the next token
#     predictions = model(input_ids)[0]
#     last_prediction = predictions[0][-1]
    
#     # Get the token with the highest probability
#     next_token = torch.argmax(last_prediction).item()
    
#     # Convert the token to its string representation
#     next_token_str = tokenizerr.decode([next_token])
    
#     # Replace the mask token with the predicted token
#     input_ids_list[-1] = next_token
    
#     # Print the generated token and the updated sequence
#     print(next_token_str, tokenizerr.decode(input_ids_list))

# Tokenize the input sentence

# Maximum length of the sequence to generate
max_length = 50

# Loop to generate new tokens until the maximum length is reached or a stopping token is generated
while True:
    # Generate the model's output for the current input sequence
    outputs = model(input_ids)
    
    # Get the last generated token's logits and indices
    logits = outputs.logits[:, -1, :]
    indices = torch.argmax(logits, dim=-1)
    
    # Get the generated token from the tokenizer
    token = tokenizerr.convert_ids_to_tokens(indices.tolist())[0]
    
    # Append the generated token to the input sequence
    input_ids = torch.cat((input_ids, indices.unsqueeze(0)), dim=1)
    
    # If the generated token is a stopping token or the maximum length is reached, break the loop
    if token in ['<eos>', '<pad>', '<s>', '<unk>'] or input_ids.shape[1] >= max_length:
        break

# Decode the final generated sequence and print it
generated_text = tokenizerr.decode(input_ids[0], skip_special_tokens=True)
print(sentence)
print(generated_text)