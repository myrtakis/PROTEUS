import json
from pathlib import Path


def save(data, folders_array, repetition, func=lambda o: o):
    path_to_folder = Path(*folders_array)
    path_to_folder.mkdir(parents=True, exist_ok=True)
    output_file = Path(path_to_folder, 'repetition' + str(repetition) + '.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, default=func, indent=4, separators=(',', ': '), ensure_ascii=False))