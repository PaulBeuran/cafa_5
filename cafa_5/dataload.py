import pandas as pd
from fasta import FASTA

def read_prot_seq_from_fasta_as_df(
    prot_seq_fasta_path : str
) -> pd.DataFrame:
    """
    Read proteins' amino-acids sequences from FASTA file
    """

    # Read FASTA file
    prot_seq_fasta = FASTA(prot_seq_fasta_path)
    
    # Extract sequences informations as dataframe
    prot_seq_data = [(seq.id, seq.name, seq.description, str(seq.seq)) 
                     for seq in prot_seq_fasta]
    prot_seq_df = pd.DataFrame(prot_seq_data, columns=["id", "name", "description", "sequence"])
    return prot_seq_df


def read_prot_go_code_from_tsv_as_dict(
    prot_go_codes_tsv_path : str
) -> pd.DataFrame:
    """
    Read proteins' GO codes from TSV file
    """
    
    # Read TSV file
    prot_go_code_df = pd.read_csv(prot_go_codes_tsv_path, sep="\t")
    prot_go_code_df.columns = ["id", "go_code", "subontology"]
    return prot_go_code_df


def read_go_code_info_accr_weight_from_txt_as_df(
    info_accr_weights_txt_path : str
) -> pd.DataFrame:
    """
    Read GO codes' information accreation (IA) F-score weights from text file
    """
    
    # Read TSV file
    info_accr_weights_df = pd.read_csv(info_accr_weights_txt_path, sep="\t", header=None)
    info_accr_weights_df.columns = ["go_code", "info_accretion_weights"]
    return info_accr_weights_df


def get_cafa_5_data_dict(
    prot_seq_fasta_path : str,
    prot_go_codes_tsv_path : str,
    info_accr_weights_txt_path : str,
):
    """
    Create a general data dictionary with CAFA 5 datas, used for models
    TODO: Read go-basic.obo to get full subontology data
    """
    
    # Load necesseary files
    prot_seq_df = read_prot_seq_from_fasta_as_df(prot_seq_fasta_path)
    prot_go_code_df = read_prot_go_code_from_tsv_as_dict(prot_go_codes_tsv_path)
    info_accr_weights_df = read_go_code_info_accr_weight_from_txt_as_df(info_accr_weights_txt_path)

    # Return data dictionary
    return {
        "prot_id": prot_seq_df["id"].tolist(),
        "prot_seq": prot_seq_df["sequence"].tolist(),
        "prot_go_codes": (prot_go_code_df.groupby("id")["go_code"]
                                         .apply(list)[prot_seq_df["id"]]
                                         .tolist()),
        "go_code_id": info_accr_weights_df["go_code"],
        "go_code_info_accr_weight": info_accr_weights_df["info_accretion_weights"],
    }
    