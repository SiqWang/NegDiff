import logging

class Logger:
    def __init__(self, filename):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def log(self, text):
        print(text)
        self.logger.info(text)
        

def merge_dicts_append(dict1, dict2):    
    for key in dict2:
        assert key in dict1.keys()        
        if isinstance(dict2[key], list):
            dict1[key].append(dict2[key])
        else:
            dict1[key] += [dict2[key]]        
    return dict1


def dummy(images, **kwargs):
    return images, False

