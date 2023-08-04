import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import time
import os

#hyperparameters
vocab_size = 3
block_size = 361
n_embd = 256
dropout = .1
n_heads = 1
n_layer = 1


directory = 'CSV/8p/'
path = 'Models/07-31.pth'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def place(s):
    return ord(s[0])-97 + (ord(s[1])-97)*19

def generate_split(directory):
    count = len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
    split = range(count)
    csv = {} 
    csv['train'] = split[0:int(0.9*count)]   
    csv['test'] = split[int(0.9*count):-1]
    return csv

def file_count(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
    
def game_from_csv(csv): #gets batch
    with open(csv + '.csv', 'r') as f:
        lines = f.readlines()
    lines = lines[1:-1]
    game_len = len(lines)
    x = torch.zeros((game_len,361),device=device,dtype=torch.int)
    y = torch.zeros((game_len,361),device=device,dtype=torch.int)
    board = torch.zeros(361,device=device,dtype=torch.int)

    
    for i,element in enumerate(lines):
        x[i] = board
        if element[3] == "B":
            board[place(element)] = 1
        elif element[3] == "W":
            board[place(element)] = 2
        y[i] = board
 
    return x,y # tensor of (game_len-1, 361) corresopnding to board states

def moves_to_tensor(moves,board=None):
    # Initialize a tensor (board)
    tensor = torch.zeros((1, 361)).long()
    if board != None:
        tensor = board
    # Define board size
    board_size = 19

    # Initialize color (1 for black, 2 for white)
    color = 1

    # Iterate over moves
    for move in moves:
        # Convert move to board index
        x = ord(move[0]) - ord('a')
        y = ord(move[1]) - ord('a')
        index = x * board_size + y

        # Set the value at the index to color
        tensor[0, index] = color

        # Switch color for the next move
        color = 3 - color

    return tensor

def input_moves(prev_board=None): #terminal input to tensor
    # Prompt user for a list of moves
    moves= input("Enter a list of moves, separated by spaces: ")

    # Split the input into a list of moves
    moves = moves.split()

    # Convert moves to tensor
    tensor = moves_to_tensor(moves,board=prev_board)
    display_board(tensor)
    return tensor

def display_board(tensor):
    # Define board size
    board_size = 19

    # Convert tensor to 19x19 grid
    grid = tensor.view(board_size, board_size)

    # Define characters for black, white, and empty
    characters = {0: '.', 1: 'B', 2: 'W'}

    # Iterate over grid and print each row
    for row in grid:
        for cell in row:
            print(characters[cell.item()], end=' ')
        print()

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size)
        self.query = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x) #(B,T,head_size)
        q = self.query(x) #(B,T,head_size)
        v = self.value(x) #(B,T,head_size)
        
        wei = q @ k.transpose(-2,-1) * C ** -0.5
        wei = F.softmax(wei,dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v
        return out        
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd,n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.net =  nn.Sequential(nn.Linear(n_embd,4*n_embd),
                    nn.ReLU(),
                    nn.Linear(4*n_embd,n_embd),
                    nn.Dropout(dropout),
                
        )
    def forward(self,x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self,n_embd,n_heads):
        super().__init__()
        self.heads = MultiHeadAttention(n_heads, n_embd // n_heads)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class Transformer(nn.Module):
  
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.ln_head = nn.Linear(n_embd,vocab_size)
        self.blocks = nn.Sequential(*[Block(n_embd,n_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.ln_head = nn.Linear(n_embd,vocab_size)
  
    def forward(self, idx, targets=None):

        tok_emb = self.token_embedding_table(idx) #(B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(block_size,device=device)) #(T,C)

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.ln_head(x) #(B,T,vocab_size)
        B,T,C = logits.shape
        logits = logits.view(B*T,C) #(batch_size, context window, vocab_size)

        loss = None
        if targets is not None:
            targets = targets.view(B*T).long()
            loss = F.cross_entropy(logits,targets)

        logits = logits.view(B,T,C) 
        return logits, loss

    def generate(self, input=None):
        board = None
        if input == None:
            board = input_moves()
        else:
            board = input
        logits,loss = self(board)
        logits = F.softmax(logits, dim=-1)
        sampled_indices = torch.multinomial(logits[0], 1).view(1,361)
        display_board(sampled_indices)
        return sampled_indices
        

