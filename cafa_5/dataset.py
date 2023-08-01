from typing import Callable

import obonet
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer

from .dataload import (read_prots_amnino_acids_from_fasta_as_df, 
                       read_go_code_info_accr_weight_from_txt_as_df,
                       read_prot_go_code_from_tsv_as_dict)


class CAFA5Dataset(torch.utils.data.Dataset): 
    """
    Dataset of the CAFA 5 competition's data, with additional transformations
    """
    
    def __init__(
        self,
        prots_amino_acids_fasta_path: str,
        prots_go_codes_tsv_path: str = None,
        go_codes_info_accr_weights_txt_path: str = None,
        go_code_graph_obo_path: str = None,
        prots_t5_embeds_npy_path: None | str = None,
        prots_protbert_embeds_npy_path: None | str = None,
        prots_esm2_embeds_npy_path: None | str = None,
    ):
        
        super().__init__()

        prots_amino_acids_df = read_prots_amnino_acids_from_fasta_as_df(prots_amino_acids_fasta_path)
        self.prots_ids = prots_amino_acids_df["id"].tolist()
        self.prots_amino_acids = prots_amino_acids_df["amino_acids"].tolist()
        
        self.go_codes = None
        self.go_codes_info_accr_weights = None
        if go_codes_info_accr_weights_txt_path is not None:
            go_codes_info_accr_weights_df = read_go_code_info_accr_weight_from_txt_as_df(
                go_codes_info_accr_weights_txt_path
            )
            self.go_codes = sorted(go_codes_info_accr_weights_df["go_code"].tolist())
            self.go_codes_info_accr_weights = go_codes_info_accr_weights_df.set_index("go_code").loc[self.go_codes, "info_accretion_weights"].tolist()
            self.prot_go_codes_transform = MultiLabelBinarizer(sparse_output=False)
            self.prot_go_codes_transform.fit([self.go_codes])
            assert (self.go_codes == self.prot_go_codes_transform.classes_).all()
        
        self.prots_go_codes = None
        if prots_go_codes_tsv_path is not None:
            prots_go_codes_df = read_prot_go_code_from_tsv_as_dict(prots_go_codes_tsv_path)
            self.prots_go_codes = (prots_go_codes_df.groupby("id")["go_code"]
                                                .apply(list)[prots_amino_acids_df["id"].tolist()]
                                                .tolist())
            
        self.go_codes_subontology = None
        if go_code_graph_obo_path is not None:
            go_code_graph = obonet.read_obo(go_code_graph_obo_path)
            self.go_codes_subontology = [go_code_graph.nodes[go_code]["namespace"] for go_code in self.go_codes]

        self.prots_t5_embeds = None
        if prots_t5_embeds_npy_path is not None:
            self.prots_t5_embeds = torch.from_numpy(np.load(prots_t5_embeds_npy_path)).float()
        self.prots_protbert_embeds = None
        if prots_protbert_embeds_npy_path is not None:
            self.prots_protbert_embeds = torch.from_numpy(np.load(prots_protbert_embeds_npy_path)).float()
        self.prots_esm2_embeds = None
        if prots_esm2_embeds_npy_path is not None:
            self.prots_esm2_embeds = torch.from_numpy(np.load(prots_esm2_embeds_npy_path)).float()         
        
        
    def __len__(
        self
    ) -> int:
        """
        Length of dataset
        """
        return len(self.prots_ids)

    
    def __getitem__(
        self, 
        idx : int
    ) -> dict[str, any]:
        """
        Get the CAFA 5 data at given protein index
        """
        prot_data = {}
        prot_id = self.prots_ids[idx]
        if self.prots_t5_embeds is not None:
            prot_data["t5_embeddings"] = self.prots_t5_embeds[idx]
            if len(prot_data["t5_embeddings"].shape) == 1:
                prot_data["t5_embeddings"] = prot_data["t5_embeddings"][None,:]
        if self.prots_protbert_embeds is not None:
            prot_data["protbert_embeddings"] = self.prots_protbert_embeds[idx]
            if len(prot_data["protbert_embeddings"].shape) == 1:
                prot_data["protbert_embeddings"] = prot_data["protbert_embeddings"][None,:]
        if self.prots_esm2_embeds is not None:
            prot_data["esm2_embeddings"] = self.prots_esm2_embeds[idx]
            if len(prot_data["esm2_embeddings"].shape) == 1:
                prot_data["esm2_embeddings"] = prot_data["esm2_embeddings"][None,:]
        prot_data_dict = {"id": prot_id, "data": prot_data}
        if self.prots_go_codes is not None:
            prot_go_codes = self.prots_go_codes[idx]
            prot_go_codes = torch.tensor(self.prot_go_codes_transform.transform([prot_go_codes]))
            prot_data_dict["go_codes"] = prot_go_codes
        return prot_data_dict


def collate_data_dict(
    batch: list[dict[str, any]]
) -> dict[str, any]:
    """
    Collate data dictionary batches into a single data dictionary
    """
    ids = []
    t5_embeddings = []
    protbert_embeddings = []
    esm2_embeddings = []
    go_codes = []
    for data in batch:
        ids.append(data["id"])
        if "go_codes" in data:
            go_codes.append(data["go_codes"])
        t5_embeddings.append(data["data"]["t5_embeddings"])
        protbert_embeddings.append(data["data"]["protbert_embeddings"])
        esm2_embeddings.append(data["data"]["esm2_embeddings"])
    batch_data_dict = {        
        "id": ids,
        "data": {
            "t5_embeddings": torch.cat(t5_embeddings),
            "protbert_embeddings": torch.cat(protbert_embeddings),
            "esm2_embeddings": torch.cat(esm2_embeddings)
        }
    }
    if len(go_codes):
        batch_data_dict["go_codes"] = torch.cat(go_codes)
    return batch_data_dict


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