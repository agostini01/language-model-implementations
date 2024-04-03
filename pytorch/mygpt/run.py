import torch
from gpt import GPT
from runner import Solution
import os

# Define the parameters
vocab_size = 104
context_length = 128
d_embed = 252
n_layers = 6
n_heads = 6
device = torch.device('cpu')

# Initialize the model and load the weights
model = GPT(vocab_size, context_length, d_embed, d_embed,
            n_layers, n_heads).to(device)
WEIGHT_PATH = os.path.join(os.path.dirname(
    __file__), '../../models/weights.pth')
model.load_state_dict(torch.load(
    WEIGHT_PATH, map_location=torch.device('cpu')))

model.compile()

model.eval()

# Define the context and the number of new characters to generate
new_chars = 100
context = torch.zeros(1, 1, dtype=torch.int64).to(device)

# Define the mapping from integers to characters
int_to_char = {0: '\n', 1: ' ', 2: '!', 3: '"', 4: '$', 5: '%', 6: '&', 7: "'",
               8: '(', 9: ')', 10: '*', 11: '+', 12: ',', 13: '-', 14: '.', 15: '/', 16: '0',
               17: '1', 18: '2', 19: '3', 20: '4', 21: '5', 22: '6', 23: '7', 24: '8', 25: '9',
               26: ':', 27: ';', 28: '?', 29: 'A', 30: 'B', 31: 'C', 32: 'D', 33: 'E', 34: 'F',
               35: 'G', 36: 'H', 37: 'I', 38: 'J', 39: 'K', 40: 'L', 41: 'M', 42: 'N', 43: 'O',
               44: 'P', 45: 'Q', 46: 'R', 47: 'S', 48: 'T', 49: 'U', 50: 'V', 51: 'W', 52: 'X',
               53: 'Y', 54: 'Z', 55: '[', 56: ']', 57: '_', 58: 'a', 59: 'b', 60: 'c', 61: 'd',
               62: 'e', 63: 'f', 64: 'g', 65: 'h', 66: 'i', 67: 'j', 68: 'k', 69: 'l', 70: 'm',
               71: 'n', 72: 'o', 73: 'p', 74: 'q', 75: 'r', 76: 's', 77: 't', 78: 'u', 79: 'v',
               80: 'w', 81: 'x', 82: 'y', 83: 'z', 84: '{', 85: '|', 86: '}', 87: 'à', 88: 'á',
               89: 'è', 90: 'é', 91: 'ë', 92: 'ñ', 93: 'ó', 94: 'ú', 95: '\u2005', 96: '–',
               97: '—', 98: '‘', 99: '’', 100: '“', 101: '”', 102: '…', 103: '\u205f'}

# Create a Solution instance and generate the text
solution = Solution()
generated_text = solution.generate(
    model, new_chars, context, context_length, int_to_char)

# Print the generated text
print(generated_text)
