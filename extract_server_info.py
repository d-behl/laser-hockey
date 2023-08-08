import glob
import numpy as np
import codecs, json 

matches_dir = {}
paths = glob.glob('RL2023HockeyTournamentClient/serverside/games/2023/8/8/*.npz')
file_path = 'RL2023HockeyTournamentClient/data.json'
for i, path in enumerate(paths):
    files = np.load(path, allow_pickle=True)
    matches_dir[i] = {}
    for j, data in enumerate(files.values()):
         matches_dir[i][j] = {str(k): str(v) for k, v in data.item().items()}

json.dump(matches_dir, codecs.open(file_path, 'w', encoding='utf-8'), 
          separators=(',', ':'), 
          sort_keys=True, 
          indent=4)