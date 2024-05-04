import pickle 
import dill
import jsonpickle
from allennlp.data import DatasetReader, TokenIndexer, Token, Instance

file_path = "./thesis_data/preprocessed/data-train.p"


def read(file_path: str):
    success = 0
    with open(file_path, 'rb') as f_in:
        unpickler = pickle.Unpickler(f_in)
        while True:
            try:
                instance = unpickler.load()
                success +=1              
                yield instance
            except EOFError:
                break
           
    print("Success", success)
          
instances = read(file_path=file_path)

for i, instance in enumerate(instances):
    print("i", i)