from transformers import AutoTokenizer

def main():
    tokenizer = AutoTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")
    print(tokenizer.tokenize("Hei, hyvää päivää pääministeri."))

if __name__ == "__main__":
    main()