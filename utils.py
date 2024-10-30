import torch
import numpy as np


# timing function
def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.
    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.
    Returns:
        float: time between start and end in seconds """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience  # number of epochs to wait before stopping if no improvement.
        self.delta = delta        # minimum change in the monitored quantity to qualify as an improvement.
        self.best_score = None    # track the best validation score
        self.early_stop = False   
        self.counter = 0
        self.best_model_state = None  # track the best  model state

    def __call__(self, val_loss, model):
        ''' updates the early stopping logic. '''
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)
        
# wrapping what happens in one epoch into one function
def train_step(model: torch.nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                eval_metric,
                device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):  
    """
        Trains a PyTorch model for a single epoch.
        Turns a target PyTorch model to training mode and then runs through 
        all of the required training steps (forward pass, loss calculation, optimizer step).
        Args:
            model: A PyTorch model to be trained.
            train_dataloader: A DataLoader instance for the model to be trained on.
            loss_fn: A PyTorch loss function to minimize.
            optimizer: A PyTorch optimizer to help minimize the loss function.
            eval_metric: An evaluation metric (e.g., accuracy_function)
            device: A target device to compute on (e.g. "cuda" or "cpu").
        Returns:
            A tuple of training loss and training evaluation metrics results.
            In the form (train_loss, train_eval_results). For example: (0.1112, 0.8743) 
    """
    
    train_epoch_loss_, train_epoch_eval_ = 0, 0  # variables to collect sum of loss/eval. results for all batches
    model.train()
    for batch_id, batch_data in enumerate(train_dataloader):
        X, y = batch_data                   # fetch minibatch
        X, y = X.to(device), y.to(device)   # push data to GPU
        
        y_logits = model(X)                 # reagent_and_mixture_logits, reagent_only_logits, mixture_only_logits
        targets = [y, torch.zeros_like(y), torch.zeros_like(y)]
        
        train_batch_loss = 0
        train_batch_eval = 0
        for logit, target in zip(y_logits, targets):
            y_preds = torch.argmax(torch.softmax(input=logit, dim=1), dim=1).type(torch.float)  
            train_batch_loss += loss_fn(logit, target)
            train_batch_eval += eval_metric(y_preds, target)
        
        # Optimizer zero grad -> Loss backward -> Optimizer step
        optimizer.zero_grad()        
        train_batch_loss.backward()  
        optimizer.step()
        
        train_epoch_loss_ += train_batch_loss.item()  # sum of losses from each batch
        train_epoch_eval_ += train_batch_eval

    # Calculate loss and accuracy per epoch 
    train_epoch_loss = train_epoch_loss_ / len(train_dataloader)  # sum of losses for all batches divided by the number of batches 
    train_epoch_eval = train_epoch_eval_ / len(train_dataloader)
    
    return train_epoch_loss, train_epoch_eval 

def validation_step(val_dataloader: torch.utils.data.DataLoader,
                    model: torch.nn.Module,
                    loss_fn: torch.nn.Module,
                    eval_metric,
                    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    val_epoch_loss_, val_epoch_eval_ = 0, 0    # variables to collect sum of loss/eval. results for all batches
    model.eval()                               # turns off various settings in model not needed for evaluation / testing
    with torch.inference_mode():               # and a couple more things behind the scenes
        for batch_id, batch_data in enumerate(val_dataloader):
            X, y = batch_data    # fetch minibatch
            X, y = X.to(device), y.to(device)  # push data to GPU
            
            y_logits = model(X)     # reagent_and_mixture_logits, reagent_only_logits, mixture_only_logits
            targets = [y] #[y, torch.zeros_like(y), torch.zeros_like(y)]
            
            
            val_batch_loss = 0
            val_batch_eval = 0
            for logit, target in zip(y_logits, targets):
                y_preds = torch.argmax(torch.softmax(input=logit, dim=1), dim=1).type(torch.float)  
                val_batch_loss += loss_fn(logit, target)
                val_batch_eval += eval_metric(y_preds, target)
            
            val_epoch_loss_ += val_batch_loss.item()   # sum of losses from each batch
            val_epoch_eval_ += val_batch_eval 
            
            # Calculate loss and accuracy per epoch 
            val_epoch_loss = val_epoch_loss_ / len(val_dataloader)  # sum of losses for all batches divided by the number of batches 
            val_epoch_eval = val_epoch_eval_ / len(val_dataloader)
        
    return val_epoch_loss, val_epoch_eval 
    
def eval_model( model: torch.nn.Module, 
                test_dataloader: torch.utils.data.DataLoader, 
                loss_fn: torch.nn.Module, 
                eval_metric,
                device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) :
    """Returns a dictionary containing the results of model predicting on data_loader.
    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on test_dataloader.
        test_dataloader (torch.utils.data.DataLoader): The target test dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        evaluation_metric: An evaluation metric (e.g., accuracy_fn) to compare the models predictions to the truth labels.
    Returns:
        (dict): Results"""
    
    test_loss_, test_eval_ = 0, 0    # variables to collect sum of loss/eval. results for all batches
    model.eval()                     # turns off various settings in model not needed for evaluation / testing
    with torch.inference_mode():     # and a couple more things behind the scenes
        for batch_id, batch_data in enumerate(test_dataloader):
            X, y = batch_data    # fetch minibatch
            X, y = X.to(device), y.to(device)  # push data to GPU
            
            y_logits = model(X)    # forward pass
            y_pred_probs = torch.softmax(input=y_logits, dim=1) 
            y_preds = torch.argmax(y_pred_probs, dim=1).type(torch.float) # Go from logits -> pred labels
            
            # Calculate loss & accuracy 
            test_batch_loss = loss_fn(input=y_logits, target=y)  # loss for one batch
            test_batch_eval = eval_metric(y_preds, y)   
            
            test_loss_ += test_batch_loss.item()   # sum of losses from each batch
            test_eval_ += test_batch_eval.item() 
            
            # Calculate loss and accuracy per epoch 
            test_loss = test_loss_ / len(test_dataloader)  # sum of losses for all batches divided by the number of batches 
            test_eval = test_eval_ / len(test_dataloader)
    
        
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": test_loss.item(),
            "model_evals": test_eval }
    
    
def criterion(labels, weighted_loss=True):
    if weighted_loss:
        counts, edges = np.histogram(labels, bins=4)
        class_weights = 1 / counts * len(labels) / 10
        # bins = np.digitize(labels, edges) - 1
        # bins = np.minimum(bins, 3)
        # class_weights = weights[bins]        
    else:
        class_weights = np.ones(4)
    class_weights = torch.from_numpy(class_weights).type(torch.float)  # numpy array -> torch tensor with float dtype
    return torch.nn.CrossEntropyLoss(weight=class_weights)