import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import time
import os
from model import Transformer

vocab_size = 3
block_size = 361
max_iters = 0 #8p size
eval_interval = 5
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 5
n_embd = 256
dropout = .1
n_heads = 1
n_layer = 1
divergence = 0.2

directory = 'CSV/8p/'
path = 'Models/07-31.pth'

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
    tensor = torch.zeros((1, 361),dtype=torch.int,device=device).long()
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

def estimate_loss(directory):
    net.to(device)
    net.eval()
    out = {}
    csv = generate_split(directory)
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        counter = 0
        for k in csv[split]:
            if counter < eval_iters:
                xb,yb = game_from_csv(directory + str(k))
                logits,loss = net(xb,yb)
                losses[counter] = loss.item()
                counter += 1
            else:
                break
        out[split] = losses.mean()
    net.train()
    return out


net = Transformer()
net.to(device)
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
if input("load model: y/n") == "y":
    net.load_state_dict(torch.load(path))
total_params = sum(p.numel() for p in net.parameters())
print(total_params)

start = time.time()
train_losses = []
test_losses = []
csv = generate_split(directory)
for iter in range(max_iters):
    x,y = game_from_csv(directory + str(iter))
    x,y = x.long(), y.long()
    logits, loss = net(x,y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if iter % eval_interval == 0:
        end = time.time()
        losses = estimate_loss(directory)
        print(f"step {iter}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}, time elapsed {end-start:.4f}")
        train_losses.append(losses['train'])
        test_losses.append(losses['test'])
        start = time.time()
        
        if abs(losses['train'] - losses['test']) > divergence * losses['test']:
            print("aborted training due to overfitting")
            break
        
torch.save(net.state_dict(),path)
end = time.time()
print("time elapsed: ", end-start)

board = torch.zeros((1,361),device=device,dtype=torch.int)
while True:
    board = net.generate(input_moves(prev_board=board))
    
