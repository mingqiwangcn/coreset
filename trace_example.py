import torch
import torch.nn as nn
import torch.optim as optim

class ExampleModel(nn.Module):
    def __init__(self):
        super(ExampleModel, self).__init__()
        D = 2
        self.fnt = nn.Linear(D, 1)

    def forward(self, X):
        scores = self.fnt(X)
        return scores.squeeze(-1)
    
class ExampleLoss(nn.Module):
    def __init__(self):
        super(ExampleLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss() 
    
    def forward(self, scores, labels):
        tensor_labels = torch.tensor(labels).float().to(scores.device)  
        loss = self.loss_fn(scores, tensor_labels)
        return loss

def get_data():
    data = []
    data.append([[1.2, 2.3], 1])
    data.append([[1.5, 1.9], 1])
    data.append([[0.8, 1.2], 1])
    data.append([[2.3, 3.5], 1])
    data.append([[3.2, 3.6], 1])
    data.append([[6.2, 8.6], 1])

    data.append([[1.6, 0.9], 0])
    data.append([[1.9, 0.9], 0])
    data.append([[0.9, 0.5], 0])
    data.append([[2.6, 1.9], 0])
    data.append([[3.0, 2.5], 0])
    data.append([[6.3, 5.6], 0])
    return data

def main():
    torch.manual_seed(0)
    model = ExampleModel()
    loss_func = ExampleLoss()
    data = get_data()
    bsz = 2
    learing_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learing_rate)
    global_steps = 0
    for epoch in range(3):
        for idx in range(0, len(data), bsz):
            global_steps += 1
            batch_data = data[idx:(idx+bsz)] 
            batch_X = torch.tensor([a[0] for a in batch_data])
            batch_Y = [a[1] for a in batch_data] 
            optimizer.zero_grad()
            scores = model(batch_X)
            loss = loss_func(scores, batch_Y)
            print('step=%d loss=%.2f' % (global_steps, loss.item()))
           
            trace(model, '1') 
            loss.backward()
            trace(model, '2')
            optimizer.step()
            trace(model, '3')

def trace(model, tag):
    for name, param in model.named_parameters():
        import pdb; pdb.set_trace()
        if param.requires_grad:
            print(name, param) 

if __name__ == '__main__':
    main()
