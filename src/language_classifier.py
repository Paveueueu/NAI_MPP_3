from gradio.exceptions import InvalidPathError

from src.single_layer_classifier import SingleLayerClassifier

class LanguageClassifier(SingleLayerClassifier):
    def __init__(self, data_path, learning_rate, bias):
        split = split_data(load_data_with_ratios(data_path))

        self.train_data_langs = split["train"]
        self.test_data_langs = split["test"]

        super().__init__(split["classes"], learning_rate, bias)


    def learn_once(self):
        self.learn(self.train_data_langs)

    def test_once(self):
        return self.test(self.test_data_langs)

    def predict_class_of_text(self, text):
        return self.predict_class({'X': count_latin_letter_ratios(text)})


def count_latin_letter_ratios(input_string):
    import unicodedata
    normalized_string = unicodedata.normalize('NFD', input_string)
    latin_string = ''.join(c for c in normalized_string if unicodedata.category(c) != 'Mn')
    latin_letters = [c.lower() for c in latin_string if c.isalpha() and 'a' <= c.lower() <= 'z']

    total_letters = len(latin_letters)
    letter_ratios = {chr(i): 0 for i in range(ord('a'), ord('z') + 1)}

    for letter in latin_letters:
        letter_ratios[letter] += 1

    if total_letters > 0:
        for letter in letter_ratios:
            letter_ratios[letter] /= total_letters

    return list(letter_ratios.values())

def split_data(data_dict):
    folder_files = {folder: list(files) for folder, files in data_dict.items()}

    shuffled = []

    while any(folder_files.values()):
        for folder in list(folder_files.keys()):
            if folder_files[folder]:
                file = folder_files[folder].pop(0)
                shuffled.append({"X": file['ratio'], "class": folder, 'name': file['name']})

    split_index = int(len(shuffled) * 0.8)
    train_data = shuffled[:split_index]
    test_data = shuffled[split_index:]

    return {"train": train_data, "test": test_data, "classes": list(data_dict.keys())}

def load_data_with_ratios(data_path):
    import os
    import random

    data_dict = {}

    if not os.path.exists(data_path):
        raise InvalidPathError(f"Katalog {data_path} nie istnieje.")

    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):
            files_ratios = []

            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                        file_ratios = count_latin_letter_ratios(file_content)
                        files_ratios.append({'ratio': file_ratios, 'name': file})

            random.shuffle(files_ratios)
            data_dict[folder.upper()] = files_ratios
    return data_dict