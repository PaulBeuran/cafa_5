from abc import ABC
from typing import Callable

import sys
import torch
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from .dataset import CAFA5Dataset, tensor_nested_dict_to_device
from .metrics import weighted_f_score


class TorchCAFA5Model(ABC, torch.nn.Module):
    """
    Abstract class of all deep models designed for the CAFA 5 competiton.
    Any model designed for the competion should inherit from this class
    and only overload the `__init__` and `forward` method

    The method already defined by the abstract class are:
    - `to`: Transfer the model to another processing device (either CPU or GPU)
    - `fit`: train the model given some training proteins data, as well as all
    other trainign options (batch size, epochs, loss function, optimizer, etc...)
    """

    def __init__(
        self
    ) -> None:
    
        # Any child class from this class must call the latter's `__init__` method 
        # before initializing any attributes (as with torch.nn.Module)
        super().__init__()

        # Set the device by default to CPU
        self.device = "cpu"

    
    def forward(
        self,
        prots_seq: any
    ) -> any:
        """
        Make a forward computation pass on one or more protein sequences input data.
        
        Method must be defined for child classes. It must also take into account, 
        when defined, the proteins sequences and go codes respective data formats 
        expected after transformations.
        """

        raise NotImplementedError()
        

    def to(
        self,
        device: str
    ) -> None:
        """
        Tranfer the model to another device, either CPU with `device="cpu"` or
        GPU with `device="cuda"`
        """

        # Switch to the specified device
        super().to(device)
        self.device = device

        
    def fit(
        self,
        train_cafa_5_dataset: CAFA5Dataset,
        epochs : int = 1,
        batch_size : int = 1,
        collate_fn : None | Callable[[any], any] = None,
        loss_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.nn.BCELoss(),
        loss_kwargs : None | dict[str, any] = None,
        optimizer_type : type = torch.optim.SGD,
        optimizer_kwargs : None | dict = None,
        validation_size : float = 0.0,
        verbose : bool = False
    ) -> None:
        """
        Train the model given some proteins data, using the back-propagation algorithm.

        The training data must be a `CAFA5Dataset` object.
        
        The user can specify as well:
            - The number of training epochs
            - The batch size for each training step in an epoch
            - The loss function, default to binary element-wise cross entropy (BCE),
            and its optional additional arguments through `loss_kwargs`
            - The optimizzer type, default to stochastic gradient descent (SGD),
            and its optional additional arguments through `optimizer_kwargs
            - The validation size, which enable the validation loop on a subset
            of trainin data at the end of each training loop for each epoch
            - The verbose option, which enables visualization of training progress,
            in both time and performance metrics
        """

        # Get information accretion weights from dataset
        go_codes_info_accr_weights = train_cafa_5_dataset.go_codes_info_accr_weights
        go_codes_info_accr_weights = torch.tensor(go_codes_info_accr_weights).to(self.device)

        # Initialize the optimizer
        optimizer_kwargs_ = optimizer_kwargs if optimizer_kwargs is not None else {} 
        optimizer = optimizer_type(self.parameters(), **optimizer_kwargs_)
        
        # Initialize other arguments
        loss_kwargs_ = loss_kwargs if loss_kwargs is not None else {}
        
        # Split into train and validation set if specified
        if validation_size:
            lengths = [(1 if validation_size < 1 else len(train_cafa_5_dataset)) - validation_size, validation_size]
            train_cafa_5_dataset, val_cafa_5_dataset = (
                split for split in 
                torch.utils.data.random_split(train_cafa_5_dataset, lengths)
            )

        # Initialize data loaders
        train_data_loader = DataLoader(train_cafa_5_dataset, batch_size = batch_size, 
                                       collate_fn = collate_fn)
        if validation_size:
            val_data_loader = DataLoader(val_cafa_5_dataset, batch_size = batch_size, 
                                         collate_fn = collate_fn)
        
        # Train loop by epochs
        for epoch in range(epochs):
            
            # Set the model to training mode
            self.train()
            
            # Initialize epoch's metrics
            train_running_loss = 0
            train_running_f_score = 0

            # Add a layer on the iterators for the progress bar
            if verbose:
                train_data_loader_ = tqdm(train_data_loader, position=0, leave=True)
                if validation_size:
                    val_data_loader_ = tqdm(val_data_loader, position=0, leave=True)
            else:
                train_data_loader_ = train_data_loader
                if validation_size:
                    val_data_loader_ = val_data_loader
            
            # Train step by bach
            for i, data_batch in enumerate(train_data_loader_):

                # Transfer the data to device
                data_batch = tensor_nested_dict_to_device(
                    data_batch, self.device
                )
                
                # Reset the gradient
                optimizer.zero_grad()

                # Make predictions
                go_codes_probs_batch = self(data_batch["sequence"])
                go_codes_preds_batch = (go_codes_probs_batch >= 0.5).float()#.to_sparse_csr()
                
                # Compute loss and backpropagate loss gradient for each parameters
                loss = loss_fn(go_codes_probs_batch, data_batch["go_codes"].float(), **loss_kwargs_)
                loss.backward()
                
                # Optimize the parameters according to the gradient and the optimizer
                optimizer.step()
                
                # Update running metrics
                train_running_loss += loss.item()
                train_running_f_score += weighted_f_score(go_codes_preds_batch, 
                                                          data_batch["go_codes"], 
                                                          go_codes_info_accr_weights)
                
                # Update the progress bar if needed
                if verbose:
                    train_data_loader_.set_description(f"- Epoch: {epoch}, Mode: train, " + 
                                                      f"Loss: {round(train_running_loss/(i+1), 6)}, " +  
                                                      f"F-score : {round(train_running_f_score/(i+1), 6)}", 
                                                      refresh=True)
               
            # If validation size has been specificed
            if validation_size:
                
                # Set the model to evaluation mode
                self.eval()
                
                # Initialize epoch's metrics
                val_running_loss = 0
                val_running_f_score = 0
                
                # No need to compute the gradients when evaluating
                with torch.no_grad():
                    
                    # Validation step by batch
                    for i, data_batch in enumerate(val_data_loader_):

                        # Transfer the data to device
                        data_batch = tensor_nested_dict_to_device(
                            data_batch, self.device
                        )
                    
                        # Make predictions
                        go_codes_probs_batch = self(data_batch["sequence"])
                        go_codes_preds_batch = (go_codes_probs_batch >= 0.5).float()
                        
                        # Compute metrics
                        val_running_loss += loss_fn(go_codes_probs_batch, data_batch["go_codes"].float()).item()
                        val_running_f_score += weighted_f_score(go_codes_preds_batch, 
                                                                data_batch["go_codes"], 
                                                                go_codes_info_accr_weights)

                        # Update the progress bar if needed
                        if verbose:
                            val_data_loader_.set_description(f"- Epoch: {epoch}, Mode: validation, " + 
                                                            f"Loss: {round(val_running_loss/(i+1), 6)}, " +  
                                                            f"F-score : {round(val_running_f_score/(i+1), 6)}", 
                                                            refresh=True)
                        

class FCNN(torch.nn.Module):
    """
    Fully connected neural network class
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        hidden_activation,
        output_activation,
        num_layers: int = 0,
        dropout: float = 0.,
    ):
        # Initialize torch.nn.Module
        super().__init__()

        # Set MLP attributes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.dropout = dropout
        
        # Create the MLP from a sequence of linear transformations followed by
        # the element-wise non-linear operation, with additional dropout if specified
        mlp = []
        for i in range(num_layers):
            mlp.append(torch.nn.Linear(input_size if not i else hidden_size, hidden_size))
            mlp.append(self.hidden_activation)
            if dropout:
                mlp.append(torch.nn.Dropout(dropout))
        mlp.append(torch.nn.Linear(input_size if num_layers == 0 else hidden_size, output_size))
        mlp.append(output_activation)
        self.mlp = torch.nn.Sequential(*mlp)
        

    def forward(self, inputs):

        # Pass the inputs to the MLP class
        return self.mlp(inputs)
            

class CAFA5LSTM(TorchCAFA5Model):
    """
    LSTM folled by an FCL, applied to CAFA 5 competion
    """

    def __init__(
        self,
        amino_acids_vocab: list[str],
        go_codes_vocab: list[str],
        embedding_kwargs: dict[str, any],
        lstm_kwargs: dict[str, any],
        fcnn_kwargs: dict[str, any]
    ):
        # Initialize TorchCAFA5Model
        super().__init__()
        
        # Initialize embedding layer
        embedding_kwargs["num_embeddings"] = len(amino_acids_vocab)
        self.embedding = torch.nn.Embedding(**embedding_kwargs)
        
        # Initialize LSTM layers
        lstm_kwargs["input_size"] = embedding_kwargs["embedding_dim"]
        lstm_kwargs["batch_first"] = True
        self.lstm = torch.nn.LSTM(**lstm_kwargs)
        
        # Initialize FCN layers
        fcnn_kwargs["input_size"] = ((self.lstm.bidirectional + 1) * 
                                    (self.lstm.proj_size if self.lstm.proj_size > 0 
                                                         else self.lstm.hidden_size))
        fcnn_kwargs["output_size"] = len(go_codes_vocab)
        fcnn_kwargs["output_activation"] = torch.nn.Sigmoid()
        self.fcnn = FCNN(**fcnn_kwargs)
        
    
    def forward(
        self,
        prot_seq
    ):
        """
        Compute the protein's GO codes probs. from their amino-acids sequence as such:
        - Get the amino-acids embeddings from the embedding layer using their respective tokens
        - Get the protein sequence embedding from the LSTM's last hidden state using the amino-acids embeddings
        - Get the GO codes probs. from the FCNN output using the the protein sequence embedding
        """
        
        # Get the amino-acids embeddings
        amino_acids_embeddings = self.embedding(prot_seq["input_ids"])

        # Pack the padded sequence, to avoid useless computations of padding
        amino_acids_embeddings_packed = torch.nn.utils.rnn.pack_padded_sequence(
            amino_acids_embeddings,
            prot_seq["attention_mask"].sum(dim=1).to("cpu"), 
            batch_first=True,
            enforce_sorted=False
        )

        # Get the protein sequence embedding from the LSTM's last hidden state
        prot_seq_embedding = self.lstm(amino_acids_embeddings_packed)[1][0]
        prot_seq_embedding = prot_seq_embedding.view(self.lstm.num_layers, 2, 
                                                     amino_acids_embeddings.shape[0], 
                                                     self.lstm.hidden_size)
        prot_seq_embedding = prot_seq_embedding[-1]
        prot_seq_embedding = torch.cat([prot_seq_embedding[0], prot_seq_embedding[1]], dim=1)
        del amino_acids_embeddings

        # Get the GO codes probs
        go_codes_preds = self.fcnn(prot_seq_embedding)
        del prot_seq_embedding
        return go_codes_preds