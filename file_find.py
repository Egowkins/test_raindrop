import os
import re


def finder():

    path = os.getcwd()

    dir_pattern = re.compile(r"^M\d+$")
    file_pattern = re.compile(r"^M\d+_\d+\.mat$")

    # Список для собранных файлов
    collected_files = []

    # Итерация по элементам в директории
    for item in os.listdir(path):
        item_path = os.path.join(path, item)

        if os.path.isdir(item_path) and dir_pattern.match(item):
            for file in os.listdir(item_path):
                if file_pattern.match(file):
                    collected_files.append(os.path.join(item_path, file))

    # Сортировка списка файлов
    collected_files.sort()
    print(collected_files)

    return collected_files

