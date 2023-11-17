# simple script for embedding papers using huggingface Specter
# requirement: pip install --upgrade transformers==4.2.2
from transformers import AutoModel, AutoTokenizer
import json
import argparse
from tqdm.auto import tqdm
import pathlib


class Dataset:

    def __init__(self, data_path, max_length=512, batch_size=32, tokenizer_name="allenai/specter"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.batch_size = batch_size
        # data is assumed to be a json file
        with open(data_path) as f:
            # key: 'paper_id', value: paper data (including 'title', 'abstract')
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def batches(self):
        # create batches
        batch = []
        batch_ids = []
        batch_size = self.batch_size
        i = 0
        for k, d in self.data.items():
            if (i) % batch_size != 0 or i == 0:
                batch_ids.append(k)
                batch.append(d['title'] + ' ' + (d.get('abstract') or ''))
            else:
                input_ids = self.tokenizer(batch, padding=True, truncation=True, 
                                           return_tensors="pt", max_length=self.max_length)
                yield input_ids.to('cpu'), batch_ids
                batch_ids = [k]
                batch = [d['title'] + ' ' + (d.get('abstract') or '')]
            i += 1
        if len(batch) > 0:
            input_ids = self.tokenizer(batch, padding=True, truncation=True, 
                                       return_tensors="pt", max_length=self.max_length)
            input_ids = input_ids.to('cpu')
            yield input_ids, batch_ids

class Model:

    def __init__(self, model_name="allenai/specter"):
        print("Using model {}".format(model_name))
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to('cpu')
        self.model.eval()

    def __call__(self, input_ids):
        output = self.model(**input_ids)
        return output.last_hidden_state[:, 0, :] # cls token

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', help='path to a json file containing paper metadata')
    parser.add_argument('--output', help='path to write the output embeddings file. '
                                        'the output format is jsonlines where each line has "paper_id" and "embedding" keys')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size for prediction')
    parser.add_argument('--model_and_tokenizer', default="allenai/specter", type=str, help="Huggingface model and tokenizer")
    args = parser.parse_args()
    model_and_tokenizer = args.model_and_tokenizer

    if model_and_tokenizer not in ["TurkuNLP/bert-base-finnish-cased-v1", "TurkuNLP/sbert-cased-finnish-paraphrase"]:
        raise Exception("Invalid model_and_tokenizer argument {}".format(model_and_tokenizer))

    print("Using model and tokenizer: '{}'".format(model_and_tokenizer))
    dataset = Dataset(data_path=args.data_path, batch_size=args.batch_size, tokenizer_name=args.model_and_tokenizer)
    model = Model(model_name=args.model_and_tokenizer)
    results = {}
    batches = []
    for batch, batch_ids in tqdm(dataset.batches(), total=len(dataset) // args.batch_size):
        batches.append(batch)
        emb = model(batch)
        for paper_id, embedding in zip(batch_ids, emb.unbind()):
            results[paper_id] = {"paper_id": paper_id, "embedding": embedding.detach().cpu().numpy().tolist()}

    print("Writing output to {}".format(args.output))
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as fout:
        for res in results.values():
            fout.write(json.dumps(res) + '\n')

    print("Saved output to {}".format(args.output))

if __name__ == '__main__':
    main()
