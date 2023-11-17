import torch
import transformers
import os
import argparse
from common import setup_logging
import pathlib
logger = setup_logging()

def main(model_name, output_dir):
        logger.info("Model name: {}".format(model_name))
        try:
            model = transformers.AutoModel.from_pretrained(model_name)
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        except:
            raise Exception("Failed loading model and tokenizer.")

        model_dir = "{}".format(model_name).replace("/", "_")
        output_dir = "{}/{}".format(output_dir, model_dir)

        if os.path.exists(output_dir):
            logger.info("Path {} exists. Not creating".format(output_dir))
        else:
            logger.info("Creating output dir {}".format(output_dir))
            pathlib.Path(output_dir).mkdir(parents=True)

        output_config_path = os.path.join(output_dir, "bert_config.json")
        output_model_path = os.path.join(output_dir, "pytorch_model.bin")
        model_to_save = model.module if hasattr(model, 'module') else model

        logger.info("Saving tokenizer vocabulary to path {}".format(output_dir))
        tokenizer.save_vocabulary(output_dir)
        logger.info("Saved")

        logger.info("Saving model config to path {}".format(output_config_path))
        model_to_save.config.to_json_file(output_config_path)
        logger.info("Saved")

        logger.info("Saving model state dict to path {}".format(output_model_path))
        torch.save(model_to_save.state_dict(), output_model_path)
        logger.info("Saved.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", default="./models")
    args = ap.parse_args()
    main(args.model, args.output)
    logger.info("Finished")