"run this script to create training pickled training files"
import logging
import os

import argparse
import json
import multiprocessing
import pathlib
import pickle
from time import time
from typing import Dict, Optional, Tuple, List, Any

import tqdm
from allennlp.common import Params
from allennlp.data import DatasetReader, TokenIndexer, Token, Instance
from allennlp.data.fields import TextField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer, PretrainedBertIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import WordSplitter, SimpleWordSplitter, BertBasicWordSplitter
from allennlp.training.util import datasets_from_params

from multiprocessing import Pool
import multiprocessing

from specter.data_utils import triplet_sampling_parallel


def init_logger(*, fn=None):

    # !!! here
    from importlib import reload # python 2.x don't need to import reload, use it directly
    reload(logging)

    logging_params = {
        'level': logging.INFO,
        'format': '%(asctime)s,%(msecs)d %(levelname)-3s [%(filename)s:%(lineno)d] %(message)s'
    }

    if fn is not None:
        logging_params['filename'] = fn

    logging.basicConfig(**logging_params)
    logging.info('reloading logger')


bert_params = {
    "do_lowercase": "false",
    "pretrained_model": "data/scivocab_scivocab_uncased/vocab.txt",
    "use_starting_offsets": "true"
}

NO_VENUE = '--no_venue--'


# ---------------
# instead of a class with its own constructor we define global variables
# and set their values using a `set_values` function
# this is not a clean design and has been done to support multiprocessing
# for more context, see: https://stackoverflow.com/questions/3288595/multiprocessing-how-to-use-pool-map-on-a-function-defined-in-a-class/21345308#21345308

# global variables
_tokenizer = None
_token_indexers = None
_token_indexer_id = None
_max_sequence_length = None
_concat_title_abstract = None
_data_source = None
_included_text_fields = None

# ----------------


def set_values(max_sequence_length: Optional[int] = -1,
               concat_title_abstract: Optional[bool] = None,
               data_source: Optional[str] = None,
               included_text_fields: Optional[str] = None
               ) -> None:
    # set global values
    # note: a class with __init__ would have been a better design
    # we have this structure for efficiency reasons: to support multiprocessing
    # as multiprocessing with class methods is slower
    global _tokenizer
    global _token_indexers
    global _token_indexer_id
    global _max_sequence_length
    global _concat_title_abstract
    global _data_source
    global _included_text_fields

    if _tokenizer is None:  # if not initialized, initialize the tokenizers and token indexers
        logger.info("Initializing tokenizers and token indexers")
        logger.info("bert_params: {}".format(bert_params))
        _tokenizer = WordTokenizer(word_splitter=BertBasicWordSplitter(do_lower_case=bert_params["do_lowercase"])) # WHAT TOKENIZER SHOULD I USE HERE?
        _token_indexers = {"bert": PretrainedBertIndexer.from_params(Params(bert_params))}
        _token_indexer_id = {"tokens": SingleIdTokenIndexer(namespace='id')}
    _max_sequence_length = max_sequence_length
    _concat_title_abstract = concat_title_abstract
    _data_source = data_source
    _included_text_fields = included_text_fields


def get_text_tokens(title_tokens, abstract_tokens, abstract_delimiter):
    """ concats title and abstract using a delimiter"""
    if title_tokens[-1] != Token('.'):
            title_tokens += [Token('.')]

    title_tokens = title_tokens + abstract_delimiter + abstract_tokens
    return title_tokens

def get_instance(paper):
    global _tokenizer
    global _token_indexers
    global _token_indexer_id
    global _max_sequence_length
    global _concat_title_abstract
    global _data_source
    global _included_text_fields

    included_text_fields = set(_included_text_fields.split())

    query_abstract_tokens = _tokenizer.tokenize(paper.get("query_abstract") or "") #SHOULD I USE FINBERT TOKENIZER HERE?
    query_title_tokens = _tokenizer.tokenize(paper.get("query_title") or "")

    pos_abstract_tokens = _tokenizer.tokenize(paper.get("pos_abstract") or "")
    pos_title_tokens = _tokenizer.tokenize(paper.get("pos_title") or "")

    neg_abstract_tokens = _tokenizer.tokenize(paper.get("neg_abstract") or "")
    neg_title_tokens = _tokenizer.tokenize(paper.get("neg_title") or "")

    if _concat_title_abstract and 'abstract' in included_text_fields:
        abstract_delimiter = [Token('[SEP]')]
        query_title_tokens = get_text_tokens(query_title_tokens, query_abstract_tokens, abstract_delimiter)
        pos_title_tokens = get_text_tokens(pos_title_tokens, pos_abstract_tokens, abstract_delimiter)
        neg_title_tokens = get_text_tokens(neg_title_tokens, neg_abstract_tokens, abstract_delimiter)
        query_abstract_tokens = pos_abstract_tokens = neg_abstract_tokens = []

    max_seq_len = _max_sequence_length

    if _max_sequence_length > 0:
        query_abstract_tokens = query_abstract_tokens[:max_seq_len]
        query_title_tokens = query_title_tokens[:max_seq_len]
        pos_abstract_tokens = pos_abstract_tokens[:max_seq_len]
        pos_title_tokens = pos_title_tokens[:max_seq_len]
        neg_abstract_tokens = neg_abstract_tokens[:max_seq_len]
        neg_title_tokens = neg_title_tokens[:max_seq_len]

    fields = {
        "source_title": TextField(query_title_tokens, token_indexers=_token_indexers),
        "pos_title": TextField(pos_title_tokens, token_indexers=_token_indexers),
        "neg_title": TextField(neg_title_tokens, token_indexers=_token_indexers),
        'source_paper_id': MetadataField(paper['query_paper_id']),
        "pos_paper_id": MetadataField(paper['pos_paper_id']),
        "neg_paper_id": MetadataField(paper['neg_paper_id']),
    }

    if not _concat_title_abstract:
        if query_abstract_tokens:
            fields["source_abstract"] = TextField(query_abstract_tokens, token_indexers=_token_indexers)
        if pos_abstract_tokens:
            fields["pos_abstract"] = TextField(pos_abstract_tokens, token_indexers=_token_indexers)
        if neg_abstract_tokens:
            fields["neg_abstract"] = TextField(neg_abstract_tokens, token_indexers=_token_indexers)

    if _data_source:
        fields["data_source"] = MetadataField(_data_source)

    return Instance(fields)

class TrainingInstanceGenerator:

    def __init__(self,
                 data,
                 metadata,
                 samples_per_query: int = 5,
                 margin_fraction: float = 0.5,
                 ratio_hard_negatives: float = 0.3,
                 data_source: str = None):
        self.samples_per_query = samples_per_query
        self.margin_fraction = margin_fraction
        self.ratio_hard_negatives = ratio_hard_negatives
        self.paper_feature_cache = {}
        self.metadata = metadata
        self.data_source = data_source

        self.data = data
        # self.triplet_generator = TripletGenerator(
        #     paper_ids=list(metadata.keys()),
        #     coviews=data,
        #     margin_fraction=self.margin_fraction,
        #     samples_per_query=self.samples_per_query,
        #     ratio_hard_negatives=self.ratio_hard_negatives
        # )

    def _get_paper_features(self, paper: Optional[dict] = None) -> \
        Tuple[List[Token], List[Token], List[Token], int, List[Token]]:
        if paper:
            paper_id = paper.get('paper_id')
            if paper_id in self.paper_feature_cache:  # This function is being called by the same paper multiple times.
                return self.paper_feature_cache[paper_id]


            references = paper.get('references')
            features = paper.get('abstract'), paper.get('title'), references
            self.paper_feature_cache[paper_id] = features
            return features
        else:
            return None, None, None
        
    def get_raw_instances(self, query_ids, subset_name=None, n_jobs=10):
        """
        Given query ids, it generates triplets and returns corresponding instances
        These are raw instances (i.e., untensorized objects )
        The output of this function is later used with DatasetConstructor.get_instance to convert raw fields to tensors

        Args:
            query_ids: list of query ids from which triplets are generated
            subset_name: Optional name to specify the subset (train, test, or val)

        Returns:
            list of instances (dictionaries)
            each instance is a dictionary containing fields corresponding to the `query`, `positive` and
                `negative` papers in the triplet: e.g., query_abstract, query_title, pos_title, neg_title, etc
        """
        logger.info('Generating triplets ...')
        count_success, count_fail = 0, 0
        # instances = []
        for triplet in triplet_sampling_parallel.generate_triplets(list(self.metadata.keys()), self.data,
                                                            self.margin_fraction, self.samples_per_query,
                                                            self.ratio_hard_negatives, query_ids,
                                                            data_subset=subset_name, n_jobs=n_jobs):
            try:
                query_paper = self.metadata[triplet[0]]
                pos_paper = self.metadata[triplet[1][0]]
                neg_paper = self.metadata[triplet[2][0]]
                count_success += 1

                # check if all papers have title and abstract (all must have title)
                failed = False
                for paper in (query_paper, pos_paper, neg_paper):
                    if not paper['title'] or (not paper['title'] and not paper['abstract']):
                        failed = True
                        break
                if failed:
                    count_fail += 1
                    continue

                query_abstract, query_title, query_refs = self._get_paper_features(query_paper)
                pos_abstract, pos_title,  pos_refs = self._get_paper_features(pos_paper)
                neg_abstract, neg_title, neg_refs = self._get_paper_features(neg_paper)

                instance = {
                    "query_abstract": query_abstract,
                    "query_title": query_title,
                    "query_paper_id": query_paper["paper_id"],
                    "pos_abstract": pos_abstract,
                    "pos_title": pos_title,
                    "pos_paper_id": pos_paper["paper_id"],
                    "neg_abstract": neg_abstract,
                    "neg_title": neg_title,
                    "neg_paper_id": neg_paper["paper_id"],
                    "data_source": self.data_source
                }
                yield instance
            except KeyError:
                # if there is no title and abstract skip this triplet
                count_fail += 1
                pass
        logger.info(f"done getting triplets, success rate:{(count_success*100/(count_success+count_fail+0.001)):.2f}%,"
                     f"total: {count_success+count_fail}")


def get_instances(data, query_ids_file, metadata, data_source=None, n_jobs=1, n_jobs_raw=12,
                  ratio_hard_negatives=0.3, margin_fraction=0.5, samples_per_query=5,
                  concat_title_abstract=False, included_text_fields='title abstract'):
    """
    Gets allennlp instances from the data file
    Args:
        data: the data file (e.g., coviews, or cocites)
        query_ids_file: train.csv file (one query paper id per line)
        metadata: a json file containing mapping between paper ids and paper dictionaries
        data_source: the source of the data (e.g., is it coviews, cite1hop, copdf?)
        n_jobs: number of jobs to process allennlp instance conversion
        n_jobs_raw: number of jobs to generate raw triplets
        ratio_hard_negatives: how many hard nagatives
        margin_fraction: margin fraction param of triplet generation
        samples_per_query: how many samples for each query paper

    Returns:
        List[Instance]
    """

    if n_jobs == 0:
        raise RuntimeError(f"argument `n_jobs`={n_jobs} is invalid, should be >0")

    generator = TrainingInstanceGenerator(data=data, metadata=metadata, data_source=data_source,
                                          margin_fraction=margin_fraction, ratio_hard_negatives=ratio_hard_negatives,
                                          samples_per_query=samples_per_query)

    set_values(max_sequence_length=512,
               concat_title_abstract=concat_title_abstract,
               data_source=data_source,
               included_text_fields=included_text_fields)

    query_ids = [line.strip() for line in open(query_ids_file)]

    instances_raw = [e for e in generator.get_raw_instances(
        query_ids, subset_name=query_ids_file.split('/')[-1][:-4], n_jobs=n_jobs_raw)]

    logger.info("n_jobs: {}".format(n_jobs))
    if n_jobs == 1:
        logger.info(f'Converting raw instances to allennlp instances:')
        for e in tqdm.tqdm(instances_raw):
            yield get_instance(e)

    else:
        logger.info(f'Converting raw instances to allennlp instances ({n_jobs} parallel jobs)')
        with Pool(n_jobs) as p:
            instances = list(tqdm.tqdm(p.imap(
                get_instance, instances_raw)))

        # multiprocessing does not work as generator, needs to generate everything
        # see: https://stackoverflow.com/questions/5318936/python-multiprocessing-pool-lazy-iteration
        return instances


def main(data_files, train_ids, val_ids, test_ids, metadata_file, outdir, n_jobs=1, njobs_raw=1,
         margin_fraction=0.5, ratio_hard_negatives=0.3, samples_per_query=5, comment='', bert_vocab='',
         concat_title_abstract=False, included_text_fields='title abstract'):
    """
    Generates instances from a list of datafiles and stores them as a stream of objects
    Args:
        data_files: list of files indicating cooccurrence counts
        train_ids: list of training paper ids corresponding to each data file
        val_ids: list of validation paper ids
        test_ids: list of test paper ids
        metadata_file: path to the metadata file (should cover all data files)
        outdir: path to an output directory
        n_jobs: number of parallel jobs for converting instances
            (in practice we did not find parallelization to help with this)
        njobs_raw: number of parallel jobs for generating triplets
            (in practice we found njobs_raw between 10 to 15 to perform fastests)
        margin_fraction: parameter for triplet generation
        ratio_hard_negatives: how many of triplets are hard negatives
        samples_per_query: how many samples per query paper
        comment: custom comment to add to the file name when saved


    Returns:
        Nothing
            Creates files corresponding to each data file
    """
    global bert_params
    logger.info("Setting pretrained_model of bert_params to {}".format(bert_vocab))
    bert_params['pretrained_model'] = bert_vocab

    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    with open(metadata_file) as f_in:
        logger.info(f'loading metadata: {metadata_file}')
        metadata = json.load(f_in)

    for data_file, train_set, val_set, test_set in zip(data_files, train_ids, val_ids, test_ids):
        logger.info(f'loading data file: {data_file}')
        with open(data_file) as f_in:
            data = json.load(f_in)
        data_source = data_file.split('/')[-1][:-5]  # e.g., coviews_v2012
        if comment:
            data_source += f'-{comment}'

        metrics = {}
        for ds_name, ds in zip(('train', 'val', 'test'), (train_set, val_set, test_set)):
            start_time = time()
            logger.info(f'Getting instances for `{data_source}` and `{ds_name}` set')
            outfile = f'{outdir}/{data_source}-{ds_name}.p'
            logger.info(f'Writing pickled output to {outfile}.')
            with open(outfile, 'wb') as f_in:
                pickler = pickle.Pickler(f_in)
                # pickler.fast = True
                idx = 0
                for instance in get_instances(data=data,
                                              query_ids_file=ds,
                                              metadata=metadata,
                                              data_source=data_source,
                                              n_jobs=n_jobs, n_jobs_raw=njobs_raw,
                                              margin_fraction=margin_fraction,
                                              ratio_hard_negatives=ratio_hard_negatives,
                                              samples_per_query=samples_per_query,
                                              concat_title_abstract=concat_title_abstract,
                                              included_text_fields=included_text_fields):
                    pickler.dump(instance)
                    idx += 1
                    # to prevent from memory blow
                    if idx % 2000 == 0:
                        pickler.clear_memo()
            metrics[ds_name] = idx
            logger.info(f'Finished writing {idx} instances to {outfile}. Time taken: {time()-start_time:.2f} seconds')
        metrics_output = f'{outdir}/{data_source}-metrics.json' 
        with open(metrics_output, 'w') as f_out2:
            logger.info("Writing metrics to file {}".format(metrics_output))
            json.dump(metrics, f_out2, indent=2)



if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', help='path to a directory containing `data.json`, `train.csv`, `dev.csv` and `test.csv` files')
    ap.add_argument('--metadata', help='path to the metadata file')
    ap.add_argument('--outdir', help='output directory to files')
    ap.add_argument('--njobs', help='number of parallel jobs for instance conversion', default=1, type=int)
    ap.add_argument('--njobs_raw', help='number of parallel jobs for triplet generation', default=12, type=int)
    ap.add_argument('--ratio_hard_negatives', default=0.3, type=float)
    ap.add_argument('--samples_per_query', default=5, type=int)
    ap.add_argument('--margin_fraction', default=0.5, type=float)
    ap.add_argument('--comment', default='', type=str)
    ap.add_argument('--data_files', help='space delimted list of data files to override default', default=None)
    ap.add_argument('--bert_vocab', help='path to bert vocab', default='data/scibert_scivocab_uncased/vocab.txt')
    ap.add_argument('--concat-title-abstract', action='store_true', default=False)
    ap.add_argument('--included-text-fields', default='title abstract', help='space delimieted list of fields to include in the main text field'
                                                                             'possible values: `title`, `abstract`, `authors`')
    args = ap.parse_args()

    data_file = os.path.join(args.data_dir, 'data.json')
    train_ids = os.path.join(args.data_dir, 'train.txt')
    val_ids = os.path.join(args.data_dir, 'val.txt')
    test_ids = os.path.join(args.data_dir, 'test.txt')

    if args.metadata:
        metadata_file = args.metadata

    init_logger()
    logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

    logger.info("data_file: {}".format(data_file))
    logger.info("train_ids: {}".format(train_ids))
    logger.info("test_ids: {}".format(test_ids))
    logger.info("val_ids: {}".format(val_ids))
    logger.info("bert_vocab path: {}".format(args.bert_vocab))
    logger.info("njobs: {}".format(args.njobs))
    logger.info("njobs_raw: {}".format(args.njobs_raw))

    start = time()
    logger.info("Starting to create training files... at start time: {}".format(start))
    main([data_file], [train_ids], [val_ids], [test_ids], metadata_file, args.outdir, args.njobs, args.njobs_raw,
         margin_fraction=args.margin_fraction, ratio_hard_negatives=args.ratio_hard_negatives,
         samples_per_query=args.samples_per_query, comment=args.comment, bert_vocab=args.bert_vocab,
         concat_title_abstract=args.concat_title_abstract, included_text_fields=args.included_text_fields
         )
    
    end = time()
    logger.info("Time taken: {} seconds ({} minutes)".format(end-start, round((end-start)/60, 2)))    
