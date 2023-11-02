from tokenizers import Tokenizer, Encoding
class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, lines, add_special_tokens=True, truncation=True, max_length=None):
        encodings = []
        for line in lines:
            encoding = self.tokenizer.encode(line)
            if truncation and max_length is not None:
                encoding.truncate(max_length)
            if add_special_tokens:
                encoding.add_special_tokens(self.tokenizer.cls_token, self.tokenizer.sep_token)
            encodings.append(encoding)
        return encodings