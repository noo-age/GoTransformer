
import sgf
import csv
import os


def sgf_to_csv(sgf_file, csv_file):
    with open(sgf_file, encoding='ISO-8859-1') as f:
        collection = sgf.parse(f.read())

    game = collection.children[0]
    moves = [(node.properties['B' if 'B' in node.properties else 'W'][0], 'B' if 'B' in node.properties else 'W') for node in game.rest]

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Move', 'Player'])  # write header
        for move in moves:
            writer.writerow(move)

directory_path = 'Pro/9p' # replace it with your directory path
all_files = os.listdir(directory_path)

count = 0
for file in all_files:
    sgf_to_csv('Pro/9p/' + file, 'CSV/' + '9p/' + str(count) + '.csv')
    count += 1

    
        
