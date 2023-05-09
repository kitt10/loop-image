from torch.optim import AdamW
import torch.nn as nn
import numpy as np
import torch


def match(hyp_, ref_, decode):
    n = 0
    ok = 0
    w_n = 0
    w_ok = 0
    for hyp, ref in zip(hyp_, ref_):
        hyp = decode(hyp)
        ref = decode(ref)
        if ref == hyp:
            ok += 1
        n += 1

        ref = ref.split()
        hyp = hyp.split()
        w_n += len(ref)
        for r1, h1 in zip(ref, hyp):
            if r1 == h1:
                w_ok += 1

    return {"SAcc": ok/n, "WAcc": w_ok/w_n, "W_N": w_n, "W_OK": w_ok, "S_N": n, "S_OK": ok, "W_Err": torch.tensor(float(w_n-w_ok), requires_grad=True), "S_Err": n-ok}


class T5FineTuner:
    
    def __init__(self, net):

        self.net = net

        self.hparams = dict(
            learning_rate=3e-4,
            adam_epsilon=1e-8,
            patience=30
        )

        self.optimizer = AdamW(self.net.model.parameters(), lr=self.hparams['learning_rate'], eps=self.hparams['adam_epsilon'])

        # Loss function
        self.criterion = self.match_criterion
        #self.criterion = nn.HingeEmbeddingLoss()

    def match_criterion(self, hyp, ref, decode):
        return match(hyp, ref, decode)['W_Err']

    def fit(self, trainloader, devloader=None, epochs=10, verbose=True):
        train_loss_list = []
        dev_loss_list = []

        for epoch in range(1, epochs+1):

            if epoch % 50 == 0:
                for g in self.optimizer.param_groups:
                    g['lr'] *= 0.9
                    print(f'Epoch {epoch}: Learning rate changed to {g["lr"]}')
            
            # Set model to train configuration
            self.net.model.train()
            epoch_train_loss_list = []
            for x, y_true in trainloader:
                
                # Clear gradient
                self.optimizer.zero_grad()

                # Make a prediction
                y_pred = self.net.model.generate(x)
                
                # Calculate loss
                #y_true[y_true[:, :] == self.net.tokenizer.pad_token_id] = -100
                loss = self.criterion(y_pred, y_true, self.net.decode)

                # Calculate gradients of parameters
                loss.backward()

                # Update parameters
                self.optimizer.step()

                epoch_train_loss_list.append(loss.data)

            # Set model to eval configuration
            self.net.model.eval()
            epoch_dev_loss_list = []
            for x, y_true in devloader:
                # Make a prediction
                y_pred = self.net.model.generate(x)
                
                # Calculate loss
                #y_true[y_true[:, :] == self.net.tokenizer.pad_token_id] = -100
                loss = self.criterion(y_pred, y_true, self.net.decode)
                print('==', loss.data, self.net.decode(y_true[0]), self.net.decode(y_pred[0]))

                epoch_dev_loss_list.append(loss.data)
            
            mean_train_loss = np.mean([l.item() for l in epoch_train_loss_list])
            mean_dev_loss = np.mean([l.item() for l in epoch_dev_loss_list])
            
            train_loss_list.append(mean_train_loss)
            dev_loss_list.append(mean_dev_loss)
            
            if verbose > 0 and epoch % 10 == 0:
                print(f'epoch {epoch}, train loss {mean_train_loss}, dev loss {mean_dev_loss}')

            if len(dev_loss_list) > self.hparams['patience'] and all([dl < mean_dev_loss for dl in dev_loss_list[-self.hparams['patience']:-1]]):
                print(f'Early stopping, dev_loss tail: {dev_loss_list[-self.hparams["patience"]:-1]}')
                break

        print(f'Final train loss: {train_loss_list[-1].item()}, dev loss: {dev_loss_list[-1].item()}')

        return train_loss_list, dev_loss_list


class T2LFineTuner:
    
    def __init__(self, net):

        self.net = net

        self.hparams = dict(
            learning_rate=0.003,
            patience=15
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = None

    def reinit_optimizer(self, parameters=None):
        if parameters:
            self.optimizer = AdamW(parameters, lr=self.hparams['learning_rate'])
        else:
            self.optimizer = AdamW(self.net.model.parameters(), lr=self.hparams['learning_rate'])

    def fit(self, trainloader, devloader=None, epochs=1000, verbose=True):
        train_loss_list = []
        dev_loss_list = []
        print(f'Finetuning {epochs} epochs...', end='\r')
        for epoch in range(1, epochs+1):

            if epoch % 50 == 0:
                for g in self.optimizer.param_groups:
                    g['lr'] *= 0.9
                    
                    if verbose:
                        print(f'Epoch {epoch}: Learning rate changed to {g["lr"]}')
            
            # Set model to train configuration
            self.net.model.train()
            epoch_train_loss_list = []
            for x, y_true, _, _ in trainloader:
                # Clear gradient
                self.optimizer.zero_grad()

                # Make a prediction
                y_pred = self.net.model(x)

                # Calculate loss
                loss = self.criterion(y_pred, y_true)

                # Calculate gradients of parameters
                loss.backward()

                # Update parameters
                self.optimizer.step()

                epoch_train_loss_list.append(loss.data)

            # Set model to eval configuration
            self.net.model.eval()
            epoch_dev_loss_list = []
            for x, y_true, _, _ in devloader:
                y_pred = self.net.model(x)

                # Calculate loss
                loss = self.criterion(y_pred, y_true)

                epoch_dev_loss_list.append(loss.data)
            
            mean_train_loss = np.mean([l.item() for l in epoch_train_loss_list])
            mean_dev_loss = np.mean([l.item() for l in epoch_dev_loss_list])
            
            train_loss_list.append(mean_train_loss)
            dev_loss_list.append(mean_dev_loss)
            
            if verbose and epoch % 10 == 0:
                print(f'epoch {epoch}, train loss {mean_train_loss}, dev loss {mean_dev_loss}')

            if len(dev_loss_list) > self.hparams['patience'] and all([dl < mean_dev_loss for dl in dev_loss_list[-self.hparams['patience']:-1]]):
                if verbose:
                    print(f'Early stopping, dev_loss tail: {dev_loss_list[-self.hparams["patience"]:-1]}')
                break

        print(f'Final train loss: {train_loss_list[-1].item()}, dev loss: {dev_loss_list[-1].item()}')

        return train_loss_list, dev_loss_list
    

class I2LFineTuner:
    
    def __init__(self, net):

        self.net = net

        self.hparams = dict(
            learning_rate=0.001,
            patience=15
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = None

    def reinit_optimizer(self, parameters=None):
        if parameters:
            self.optimizer = AdamW(parameters, lr=self.hparams['learning_rate'])
        else:
            self.optimizer = AdamW(self.net.model.parameters(), lr=self.hparams['learning_rate'])

    def fit(self, trainloader, devloader=None, epochs=10, verbose=True):
        train_loss_list = []
        dev_loss_list = []
        print(f'Finetuning {epochs} epochs...', end='\r')
        for epoch in range(1, epochs+1):

            if epoch % 50 == 0:
                for g in self.optimizer.param_groups:
                    g['lr'] *= 0.9
                    
                    if verbose:
                        print(f'Epoch {epoch}: Learning rate changed to {g["lr"]}')
            
            # Set model to train configuration
            self.net.model.train()
            epoch_train_loss_list = []
            for x, y_true, _, _ in trainloader:
                # Clear gradient
                self.optimizer.zero_grad()

                # Make a prediction
                y_pred = self.net.model(x)

                # Calculate loss
                loss = self.criterion(y_pred, y_true)

                # Calculate gradients of parameters
                loss.backward()

                # Update parameters
                self.optimizer.step()

                epoch_train_loss_list.append(loss.data)

            # Set model to eval configuration
            self.net.model.eval()
            epoch_dev_loss_list = []
            for x, y_true, _, _ in devloader:
                y_pred = self.net.model(x)

                # Calculate loss
                loss = self.criterion(y_pred, y_true)

                epoch_dev_loss_list.append(loss.data)
            
            mean_train_loss = np.mean([l.item() for l in epoch_train_loss_list])
            mean_dev_loss = np.mean([l.item() for l in epoch_dev_loss_list])
            
            train_loss_list.append(mean_train_loss)
            dev_loss_list.append(mean_dev_loss)
            
            if verbose and epoch % 1 == 0:
                print(f'epoch {epoch}, train loss {mean_train_loss}, dev loss {mean_dev_loss}')

            if len(dev_loss_list) > self.hparams['patience'] and all([dl < mean_dev_loss for dl in dev_loss_list[-self.hparams['patience']:-1]]):
                if verbose:
                    print(f'Early stopping, dev_loss tail: {dev_loss_list[-self.hparams["patience"]:-1]}')
                break

        print(f'Final train loss: {train_loss_list[-1].item()}, dev loss: {dev_loss_list[-1].item()}')

        return train_loss_list, dev_loss_list