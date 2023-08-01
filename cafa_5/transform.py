from abc import ABC
from typing import Any

import scipy

from sklearn.preprocessing import MultiLabelBinarizer

from tokenizers import Tokenizer
from tokenizers.trainers import WordLevelTrainer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast

import torch

class TrainableTransform(ABC):
    """
    Abstract class of a trainable transform
    """

    def __init__(
        self,
    ):
        self.is_fitted = False


    def fit(
        self,
        inputs: any
    ) -> None:
        """
        Fit the transform to some train_data
        """
        raise NotImplementedError()
    

    def __call__(
        self, 
        input: any,
    ) -> Any:
        """
        Transformation of the input
        """
        return NotImplementedError()
    

class CharTokenizer(TrainableTransform):
    """
    Tokenize characters using Hugging Face's fast tokenizer
    """

    def __init__(
        self,
        max_size: None | int = None,
        padding: None | str = None,
    ):
        super().__init__()
        self.tokenizer = Tokenizer(WordLevel())
        self.trainer = WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]"])
        self.tokenizer.pre_tokenizer = Split("", "removed")
        self.vocab = None


    def fit(
        self,
        char_seqs: list[str]
    ):
        """
        Fit the tokenizer vocabulary from character sequences, with the addition of
        padding and unknown tokens
        """
        self.tokenizer.train_from_iterator(char_seqs, self.trainer)
        self.vocab = self.tokenizer.get_vocab()
        if not self.max_size:
            self.max_size = max([len(seq) for seq in char_seqs])
        self.fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object = self.tokenizer, 
            model_max_length = self.max_size,
            pad_token = "[PAD]",
            unk_token = "[UNK]",
            cls_token = "[CLS]"
        )
        self.is_fitted = True
        

    def __call__(
        self,
        char_seqs: str | list[str]
    ) -> dict[str, torch.IntTensor]:
        """
        Transform the tokens into their ids. in the vocabulary, as well as the attention
        mask on the tokens
        """
        return self.fast_tokenizer(char_seqs, padding = self.padding, truncation = True, 
                                   return_tensors = "pt", return_token_type_ids = False)


def scipy_csr_to_torch_csr(
    scipy_csr: scipy.sparse.csr_array
):
    
    crow_indices = scipy_csr.indptr
    col_indices = scipy_csr.indices
    values = scipy_csr.data
    return torch.sparse_csr_tensor(crow_indices, col_indices, values)


class MultiOutputBinarizer(TrainableTransform):
    """
    Binarize a list of classes
    """

    def __init__(
        self,
        outputs: None | list[str] = None,
        sparse_output: bool = False
    ):
        super().__init__()
        self.sparse_output = sparse_output
        self.multi_output_binarizer = MultiLabelBinarizer(sparse_output=sparse_output)
        if outputs:
            self.outputs = sorted(outputs)
            self.multi_output_binarizer.fit([self.outputs])
            self.is_fitted = True
        

    def fit(
        self,
        outputs_lists: list[list[str]],
    ) -> None:
        """
        Learn about the classes to binarize from output data
        """
        self.outputs = sorted(set([output for outputs_list in outputs_lists 
                                   for output in outputs_list]))
        self.multi_output_binarizer.fit([self.outputs])
        self.is_fitted = True
    

    def __call__(
        self,
        outputs : list[str] | list[list[str]]
    ) -> torch.IntTensor:
        """
        Binarize a list of classes
        """
        if isinstance(outputs, list) and isinstance(outputs[0], str):
            outs_ = [outputs]
        else:
            outs_ = outputs
        if not self.sparse_output:
            return torch.tensor(self.multi_output_binarizer.transform(outs_))
        else:
            return scipy_csr_to_torch_csr(self.multi_output_binarizer.transform(outs_)).int()    