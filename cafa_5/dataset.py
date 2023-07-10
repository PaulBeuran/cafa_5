from typing import Callable

import torch

from .dataload import get_cafa_5_data_dict
from .transform import TrainableTransform


class CAFA5Dataset(torch.utils.data.Dataset): 
    """
    Dataset of the CAFA 5 competition's data, with additional transformations
    """
    
    def __init__(
        self,
        prot_seq_fasta_path: str,
        prot_go_codes_tsv_path: str,
        info_accr_weights_txt_path: str,
        sequence_transform: None | Callable[[str | list[str]], any] = None,
        go_codes_transform: None | Callable[[str | list[str]], any] = None
    ):
    
        # Get CAFA 5 data dictionary
        cafa_5_data_dict = get_cafa_5_data_dict(
            prot_seq_fasta_path,
            prot_go_codes_tsv_path,
            info_accr_weights_txt_path
        )

        # Extract data from CAFA 5 dictionary
        self.prots_ids = cafa_5_data_dict["prot_id"]
        self.prots_seqs = cafa_5_data_dict["prot_seq"]
        self.prots_go_codes = cafa_5_data_dict["prot_go_codes"]
        self.go_codes_ids = cafa_5_data_dict["go_code_id"]
        self.go_codes_info_accr_weights = cafa_5_data_dict["go_code_info_accr_weight"]

        # Set transforms
        self.seq_transform = sequence_transform
        self.go_codes_transform = go_codes_transform
        
        
    def __len__(
        self
    ) -> int:
        """
        Length of dataset
        """
        return len(self.prots_ids)
    

    def fit(
        self,
        fit_seq_transform: bool = True,
        fit_go_codes_transform: bool = True
    ) -> None:
        """
        Fit the transforms to data if nee
        """
        if (fit_seq_transform and 
            isinstance(self.seq_transform, TrainableTransform) and
            not self.seq_transform.is_fitted):
            self.seq_transform.fit(self.prots_seqs)

        if (fit_go_codes_transform and 
            isinstance(self.go_codes_transform, TrainableTransform) and
            not self.go_codes_transform.is_fitted):
            self.go_codes_transform.fit([self.go_codes_ids])

    
    def __getitem__(
        self, 
        idx : int
    ) -> dict[str, any]:
        """
        Get the CAFA 5 data at given protein index
        """
        id = self.prots_ids[idx]
        seq = self.prots_seqs[idx]
        if self.seq_transform is not None:
            seq = self.seq_transform(seq)
        go_codes = self.prots_go_codes[idx]
        if self.go_codes_transform is not None:
            go_codes = self.go_codes_transform(go_codes)
        return {"id": id, "sequence": seq, "go_codes": go_codes}
    

def collate_tok_mult_out_batch(
    batch: list[dict[str, any]]
) -> dict[str, any]:
    """
    Collate data dictionary batches into a single data dictionary
    """
    ids = []
    sequences_input_ids = []
    sequences_attention_mask = []
    go_codes = []
    for data in batch:
        ids.append(data["id"])
        sequences_input_ids.append(data["sequence"]["input_ids"])
        sequences_attention_mask.append(data["sequence"]["attention_mask"])
        go_codes.append(data["go_codes"]) 
    return {
        "id": ids,
        "sequence": {
            "input_ids": torch.cat(sequences_input_ids),
            "attention_mask": torch.cat(sequences_attention_mask)
        },
        "go_codes": torch.cat(go_codes)
    }


def tensor_nested_dict_to_device(
    tensor_dict: dict[str, any],
    device: str
) -> dict[str, any]:
    
    return {
        k:(v.to(device) if isinstance(v, torch.Tensor) else
           tensor_nested_dict_to_device(v, device) if isinstance(v, dict) else
           v)
        for k,v in tensor_dict.items()
    }
    