from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer


def main():
    tokenizer = PretrainedTransformerTokenizer(model_name="TurkuNLP/bert-base-finnish-cased-v1", do_lowercase=False)
    print(tokenizer.tokenize("Hei, hyvää päivää pääministeri."))

if __name__ == "__main__":
    main()