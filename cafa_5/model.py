from abc import ABC
from typing import Callable

from copy import deepcopy
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import CAFA5Dataset, tensor_nested_dict_to_device
from .metrics import weighted_f_score, weighted_f_score_by_group


class ClassWeightedBCELoss():

    def __init__(
        self,
        class_weights: torch.Tensor,
        **bce_loss_kwargs: dict[str, any]
    ) -> None:
        
        self.class_weights = class_weights
        self.reduction = "mean"
        bce_loss_kwargs["reduction"] = "none"
        self.bce_loss = torch.nn.BCELoss(**bce_loss_kwargs)

    
    def __call__(
        self,
        input: torch.Tensor,
        target: torch.Tensor
    ):
        return (self.class_weights * self.bce_loss(input, target)).mean()
    

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
        cafa_5_dataset: CAFA5Dataset,
        epochs : int = 1,
        batch_size : int = 1,
        collate_fn : None | Callable[[any], any] = None,
        loss_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.nn.BCELoss(),
        loss_kwargs : None | dict[str, any] = None,
        optimizer_type : type = torch.optim.SGD,
        optimizer_kwargs : None | dict = None,
        lr_scheduler_type : None | type = None,
        lr_scheduler_kwargs : None | dict[str, any] = None,
        validation_size : float = 0.0,
        verbose : bool = False,
        checkpoint_save_folder_path: None | str = None
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
        go_codes_info_accr_weights = cafa_5_dataset.go_codes_info_accr_weights
        go_codes_info_accr_weights = torch.tensor(go_codes_info_accr_weights).to(self.device)
        go_codes_subontology = cafa_5_dataset.go_codes_subontology

        # Initialize the optimizer
        optimizer_kwargs_ = optimizer_kwargs if optimizer_kwargs is not None else {} 
        optimizer = optimizer_type(self.parameters(), **optimizer_kwargs_)
        
        # Initialize other arguments
        loss_kwargs_ = loss_kwargs if loss_kwargs is not None else {}
        if lr_scheduler_type is not None:
            lr_scheduler = lr_scheduler_type(optimizer, **lr_scheduler_kwargs)
        
        # Split into train and validation set if specified
        if validation_size:
            lengths = [(1 if validation_size < 1 else len(cafa_5_dataset)) - validation_size, validation_size]
            train_cafa_5_dataset, val_cafa_5_dataset = (
                split for split in 
                torch.utils.data.random_split(cafa_5_dataset, lengths)
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
                go_codes_probs_batch = self(data_batch["data"])
                go_codes_preds_batch = (go_codes_probs_batch >= 0.5).float()
                
                # Compute loss and backpropagate loss gradient for each parameters
                loss = loss_fn(go_codes_probs_batch, data_batch["go_codes"].float(), **loss_kwargs_)
                loss.backward()
                
                # Optimize the parameters according to the gradient and the optimizer
                optimizer.step()

                if lr_scheduler_type is not None:
                    lr_scheduler.step()
                
                # Update running metrics
                train_running_loss += loss.item()
                train_running_f_score += weighted_f_score_by_group(go_codes_preds_batch, 
                                                                   data_batch["go_codes"], 
                                                                   go_codes_info_accr_weights,
                                                                   go_codes_subontology)
                
                # Update the progress bar if needed
                if verbose:
                    train_data_loader_.set_description(f"- Epoch: {epoch}, Mode: train, " + 
                                                      f"Loss: {round(train_running_loss/(i+1), 6)} ({round(loss.item(), 6)})," +  
                                                      f"""F-score : {round(train_running_f_score/(i+1), 6)} ({round(weighted_f_score(go_codes_preds_batch, 
                                                          data_batch["go_codes"], 
                                                          go_codes_info_accr_weights), 6)})""", 
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
                        go_codes_probs_batch = self(data_batch["data"])
                        go_codes_preds_batch = (go_codes_probs_batch >= 0.5).float()
                        
                        # Compute metrics
                        val_running_loss += loss_fn(go_codes_probs_batch, data_batch["go_codes"].float()).item()
                        val_running_f_score += weighted_f_score_by_group(go_codes_preds_batch, 
                                                                         data_batch["go_codes"], 
                                                                         go_codes_info_accr_weights,
                                                                         go_codes_subontology)

                        # Update the progress bar if needed
                        if verbose:
                            val_data_loader_.set_description(f"- Epoch: {epoch}, Mode: validation, " + 
                                                            f"Loss: {round(val_running_loss/(i+1), 6)}, " +  
                                                            f"""F-score : {round(val_running_f_score/(i+1), 6)}""", 
                                                            refresh=True)
                            
            checkpoint_save_file_path = os.path.join(checkpoint_save_folder_path, f"weights_{epoch}.pt")
            torch.save({
                "epoch": epoch, 
                "state_dict": self.state_dict(), 
                "train_loss": train_running_loss/len(train_data_loader_),
                "train_f_score": train_running_f_score/len(train_data_loader_),
                "val_loss": val_running_loss/len(val_data_loader_),
                "val_f_score": val_running_f_score/len(val_data_loader_),
            }, checkpoint_save_file_path)
                            

    def predict_proba(
        self,
        cafa_5_dataset: CAFA5Dataset,
        batch_size : int = 1,
        collate_fn : None | Callable[[any], any] = None,
        verbose: bool = False,        
        submission_tsv_path: str = "submission.tsv"
    ):
        
        # Initialize data loaders
        data_loader = DataLoader(cafa_5_dataset, batch_size = batch_size, 
                                 collate_fn = collate_fn)
        # Set the model to evaluation mode
        self.eval()

        # Add a layer on the iterators for the progress bar
        if verbose:
            data_loader = tqdm(data_loader, position=0, leave=True)

        # No need to compute the gradients when evaluating
        with torch.no_grad():
            
            with open(submission_tsv_path, "w+") as f:
                
                f.write("Protein Id\tGO Term Id\tPrediction\n")

                # Validation step by batch
                for i, data_batch in enumerate(data_loader):

                    # Transfer the data to device
                    data_batch = tensor_nested_dict_to_device(
                        data_batch, self.device
                    )
                
                    # Make predictions
                    go_codes_probs_batch = self(data_batch["data"])
                    go_codes_pos_mask = (go_codes_probs_batch >= 0.5).cpu()
                    prots_go_codes = cafa_5_dataset.prot_go_codes_transform.inverse_transform(go_codes_pos_mask)
                    for i in range(go_codes_probs_batch.shape[0]):
                        prot_id = data_batch["id"][i]
                        prot_go_codes = prots_go_codes[i]
                        prot_go_codes_probs = go_codes_probs_batch[i, go_codes_pos_mask[i]]
                        for j in range(len(prot_go_codes)):
                            prot_go_code = prot_go_codes[j]
                            prot_go_code_prob = prot_go_codes_probs[j]
                            f.write(prot_id + "\t" + prot_go_code + "\t" + str(prot_go_code_prob.item()) + "\n")


class FFN(torch.nn.Module):
    """
    Fully connected neural network class
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        output_activation: Callable[[torch.Tensor], torch.Tensor],
        num_layers: int = 0,
        hidden_size: None | int = None,
        hidden_activation: None | Callable[[torch.Tensor], torch.Tensor] = None,
        dropout: float = 0.,
        batch_normalization: bool = False,
        residual_connections: bool = False
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
        self.batch_normalization = batch_normalization
        self.residual_connections = residual_connections

        self.ffn = torch.nn.ModuleDict()
        i = -1
        if batch_normalization:
            self.bn = torch.nn.BatchNorm1d(self.input_size)
        for i in range(num_layers):
            self.ffn[f"linear_{i}"] = torch.nn.Linear(input_size if not i else hidden_size, hidden_size)
            if batch_normalization:
                self.ffn[f"batch_norm_{i}"] = deepcopy(torch.nn.BatchNorm1d(self.hidden_size))
            self.ffn[f"activation_{i}"] = deepcopy(self.hidden_activation)
            if dropout:
                self.ffn[f"dropout_{i}"] = torch.nn.Dropout(dropout)
        self.ffn[f"linear_{i+1}"] = torch.nn.Linear(input_size if num_layers == 0 else hidden_size, output_size)
        self.ffn[f"activation_{i+1}"] = output_activation


    def forward(self, input):

        output = input
        if self.batch_normalization:
            output = self.bn(output)
        for i in range(self.num_layers + 1):
            if i != 0 and i != self.num_layers and self.residual_connections:
                res = output
            output = self.ffn[f"linear_{i}"](output)
            if i != self.num_layers and self.batch_normalization:
                output = self.ffn[f"batch_norm_{i}"](output)
            output = self.ffn[f"activation_{i}"](output)
            if i != 0 and i != self.num_layers and self.residual_connections:
                output = output + res
            if i != self.num_layers and self.dropout:
                output = self.ffn[f"dropout_{i}"](output) 
        return output
    

class CAFA5EmbeddingsFFN(TorchCAFA5Model):

    def __init__(
        self,
        n_go_codes: int,
        t5_embeddings_size: int = 1024,
        protbert_embeddings_size: int = 1024,
        esm2_embeddings_size: int = 2560,
        **kwargs
    ) -> None:
        
        super().__init__()
        kwargs["input_size"] = t5_embeddings_size + protbert_embeddings_size + esm2_embeddings_size
        kwargs["output_activation"] = torch.nn.Sigmoid()
        kwargs["output_size"] = n_go_codes
        self.ffn = FFN(**kwargs)

    def forward(self, prots_seq: any) -> any:
        
        return self.ffn(torch.cat([embeddings for data_id, embeddings in prots_seq.items()
                                   if data_id not in ["amino_acids_tokens"]], dim=-1))
    

class StepLRScheduler():

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int = 512,
        warmum_steps: int = 4000
    ):
        
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmum_steps
        self.step_num = 1

    def step(
        self,
    ):
        self.optimizer.lr = self.d_model**(-0.5) * min(self.step_num**(-0.5), 
                                                       self.step_num * self.warmup_steps**(-1.5))
        self.step_num += 1