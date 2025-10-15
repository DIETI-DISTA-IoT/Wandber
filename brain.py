from modules import MLP
import torch.optim as optim
import torch.nn as nn
import torch
from threading import Lock

class Brain:

    def __init__(self, **kwargs):
        self.model = MLP(**kwargs)
        self.model.initialize_weights(kwargs.get('initialization_strategy'))
        optim_class_name = kwargs.get('optimizer', 'Adam')
        self.optimizer = getattr(optim, optim_class_name)(self.model.parameters(), lr=float(kwargs.get('learning_rate', 0.001)))
        self.loss_function = nn.BCELoss()
        self.device = torch.device(kwargs.get('device', 'cpu'))
        self.model.to(self.device)
        self.model_lock = Lock()
        self.model_saving_path = kwargs.get('model_saving_path', 'default_model.pth')


    def train_step(self, x, y):
        with self.model_lock:
            self.model.train()
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.loss_function(y_pred, y)
            loss.backward()
            self.optimizer.step()
            return y_pred.detach(), loss.item()
    

    def save_model(self):
        with self.model_lock:
            torch.save(self.model.state_dict(), self.model_saving_path)