# Pytorch packages
import torch
from torch import nn, optim, tensor
if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# sheduler
from transformers import get_linear_schedule_with_warmup

# TAPE
from tape import ProteinBertModel

# time it
import time
import numpy as np

class NNClassifier(nn.Module):
    """
    1-layer Dense Neural Net. Model for the classification task
    (benchmark model)
    """
    def __init__(self, nbLabels, nnInputDim, nnHiddenSize=200, freezeBaseModel=True):
        """
        INPUTS
        nbLabels (uint): Number of classes to be predicted
        nnInputDim: input dimension of the dense NN
        nnHiddenSize (uint): Number of units in the hidden layer of the dense NN
        freezeBaseModel (bool): Set False to fine-tune the BERT-base model; default to True
        
        Class params
        classifier: a torch.nn.Module classifier
        bert: a pretrained BERT-base protein language model
        """
        super().__init__()
        
        self.nnInputDim = nnInputDim
        self.nnHiddenSize = nnHiddenSize
        self.outputDim = nbLabels

        # Instantiate BERT-base protein language model
        self.bert = ProteinBertModel.from_pretrained('bert-base')

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.nnInputDim, self.nnHiddenSize),
            nn.ReLU(),
            nn.Linear(self.nnHiddenSize, self.outputDim)
        )

        # Freeze the BERT model
        if freezeBaseModel:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def represent_sequence(self, tokenIdx):
        """
        represent a given sequence by a single vector
        """
        pass

    def forward(self, tokenIdx):
        """
        Feed input to BERT and the classifier to compute logits.
        INPUT
        tokenIdx (torch.Tensor): an input tensor of shape (batchSize, MAX_LEN)
        
        OUTPUT
        res (torch.Tensor): an output tensor with shape (batchSize, nbLabels)
        """
        sequenceRepresentation = self.represent_sequence(tokenIdx)
        # Feed input to classifier to compute prediction result
        res = self.classifier(sequenceRepresentation)

        return res
    
    
class MeanNNClassifier(NNClassifier):
    """
    1-layer Dense Neural Net. Model for the classification task
    
    Sequence representation using mean of all word embeddings
    (benchmark model)
    """
    def __init__(self, nbLabels, nnInputDim=768, nnHiddenSize=200, freezeBaseModel=True):
        """
        INPUTS
        nbLabels (uint): Number of classes to be predicted
        nnInputDim: input dimension of the dense NN
        nnHiddenSize (uint): Number of units in the hidden layer of the dense NN
        freezeBaseModel (bool): Set False to fine-tune the BERT-base model; default to True
        
        Class params
        classifier: a torch.nn.Module classifier
        bert: a pretrained BERT-base protein language model
        """
        super().__init__(nbLabels, nnInputDim, nnHiddenSize, freezeBaseModel)
        # Specify hidden size of BERT-base model (=768)
        
        
    def represent_sequence(self, tokenIdx):
        # Feed input to BERT => embedded sequence of shape (batch size, sequence length + 2, 768)
        bertOutput = self.bert(tokenIdx)[0]
        # Use the average of the embedding to represent the whole sequence
        # (simple benchmark representation)
        # output shape: (batch size, 768)
        return bertOutput.mean(axis=1)

class LstmNNClassifier(NNClassifier):
    """
    1-layer Dense Neural Net. Model for the classification task
    
    Sequence representation using the last hidden layer of an LSTM model
    """
    def __init__(self, nbLabels, lstmHiddenSize=300, nnInputDim=300, nnHiddenSize=200,freezeBaseModel=True):
        """
        INPUTS
        nbLabels (uint): Number of classes to be predicted
        lstmHiddenSize (uint): Hidden state dimension of the LSTM model
        nnInputDim: input dimension of the dense NN
        nnHiddenSize (uint): Number of units in the hidden layer of the dense NN
        freezeBaseModel (bool): Set False to fine-tune the BERT-base model; default to True
        
        Class params
        classifier: a torch.nn.Module classifier
        bert: a pretrained BERT-base protein language model
        """
        if nnInputDim != lstmHiddenSize:
            nnInputDim = lstmHiddenSize  # because the lstm uses bias
        super().__init__(nbLabels, nnInputDim, nnHiddenSize, freezeBaseModel) 
        self.lstmHiddenSize = lstmHiddenSize 
        
    def represent_sequence(self, tokenIdx):
        # Feed input to BERT => embedded sequence of shape (batch size, sequence length + 2, 768)
        bertOutput = self.bert(tokenIdx)[0]
        representationModel = nn.Sequential(
            # 768 = embedding dim of bert
            nn.LSTM(768, self.lstmHiddenSize, num_layers=1, bias=True, bidirectional=False, batch_first=True)
        )
        _,(representation,_) = representationModel(bertOutput) # shape(representation) = (1, batch size, hidden state size)
        # Use the last hidden state of the lstm network
        # output shape: (batch size, hidden state size)
        return representation[0] 
    
class CnnNNClassifier(NNClassifier):
    """
    1-layer Dense Neural Net. Model for the classification task
    
    Sequence representation using 1D-CNN + Max-pooling model
    """
    def __init__(self, nbLabels, sequenceLen, nbKernels=300, kernelSize=3, nnInputDim=300, nnHiddenSize=200,freezeBaseModel=True):
        """
        INPUTS
        nbLabels (uint): Number of classes to be predicted
        nbKernels (uint): Number of kernels (output channels)
        kernelSize (uint): Size of the kernels
        nnInputDim: input dimension of the dense NN
        nnHiddenSize (uint): Number of units in the hidden layer of the dense NN
        freezeBaseModel (bool): Set False to fine-tune the BERT-base model; default to True
        
        Class params
        classifier: a torch.nn.Module classifier
        bert: a pretrained BERT-base protein language model
        """
        # no special padding(=0), nor dilation(=1), nor stride(=1); +2 because special tokens are added to the sequence embedding
        nnInputDim = sequenceLen - (kernelSize - 1) + 2 
        super().__init__(nbLabels, nnInputDim, nnHiddenSize, freezeBaseModel) 
        self.outChannels = nbKernels
        self.kernelSize = kernelSize
        
    def represent_sequence(self, tokenIdx):
        # Feed input to BERT => embedded sequence of shape (batch size, sequence length + 2, 768)
        bertOutput = self.bert(tokenIdx)[0]
        representationModel = nn.Sequential(
            nn.Conv1d(768, self.outChannels, self.kernelSize,padding='valid')
        )
        preRepresentation = representationModel(torch.transpose(bertOutput,1,2)) # shape(representation) = (batch size, nb channels, reduced length of the input sequence)
        # max-pool over all channels
        representation,_ = preRepresentation.max(axis=1)
        # output shape: (batch size, reduced length of the input sequence)
        return representation

def initialize_classifier(classifier, batchSize, epochs=3):
    """
    Initialize a classifier model, the optimizer and the learning rate scheduler 
    that are used for training the model.
    """
    
    # where to run the model
    classifier.to(device)
    # create the optimizer
    optimizer = optim.AdamW(classifier.parameters())
    # total number of training steps
    totalSteps = batchSize * epochs
    # set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=totalSteps)
    return classifier, optimizer, scheduler

def train_model(model, optimizer, scheduler, trainData, valData, epochs=3):
    """
    Train model.
    """
    compute_cross_entropy = nn.CrossEntropyLoss()
    # Start training loop
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0Epoch, t0Batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        totalLoss, batchLoss, batchCounts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(trainData):
            batchCounts +=1
            batchIdx, batchLabels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return res.
            res = model(batchIdx)

            # Compute loss and accumulate the loss values
            loss = compute_cross_entropy(res, batchLabels)
            batchLoss += loss.item()
            totalLoss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 10 == 0 and step != 0) or (step == len(trainData) - 1):
                # Calculate time elapsed for 20 batches
                timeElapsed = time.time() - t0Batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batchLoss / batchCounts:^12.6f} | {'-':^10} | {'-':^9} | {timeElapsed:^9.2f}")

                # Reset batch tracking variables
                batchLoss, batchCounts = 0, 0
                t0Batch = time.time()

        # Calculate the average loss over the entire training data
        avgTrainLoss = totalLoss / len(trainData)
        print("-"*70)
        
        # =======================================
        #               Evaluation
        # =======================================
        valLoss, valAccuracy = evaluate_model(model, valData)

        # Print performance over the entire training data
        timeElapsed = time.time() - t0Epoch
        print(f"{epoch_i + 1:^7} | {'-':^7} | {avgTrainLoss:^12.6f} | {valLoss:^10.6f} | {valAccuracy:^9.2f} | {timeElapsed:^9.2f}")
        print("-"*70)
        print("\n")
    
    print("Training complete!")

def evaluate_model(model, valData):
    """
    Measure the model's performance
    """
    compute_cross_entropy = nn.CrossEntropyLoss()
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    valAccuracy = []
    valLoss = []

    # For each batch in our validation set...
    for batch in valData:
        # Load batch to GPU
        batchIdx, batchLabels = tuple(t.to(device) for t in batch)

        # Compute res
        with torch.no_grad():
            res = model(batchIdx)

        # Compute loss
        loss = compute_cross_entropy(res, batchLabels)
        valLoss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(res, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == batchLabels).cpu().numpy().mean() * 100
        valAccuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    valLoss = np.mean(valLoss)
    valAccuracy = np.mean(valAccuracy)

    return valLoss, valAccuracy