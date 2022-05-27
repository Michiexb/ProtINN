import torch
import torch.nn as nn



# CREATE 1-LAYER NETWORK FOR CALCULATING PROTOTYPE-CLASS WEIGHTS
class PredictionLayer(nn.Module):
    def __init__(self, input_size, num_classes):
        super(PredictionLayer, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes, bias=False) # TODO set bias to True?

    def forward(self, x):
        out = self.fc1(x)
        return out


def eval_loss(eval_loader, model, criterion, Vars):
    idm = torch.eye(len(Vars.class_codes))
    class_ids = {l: i for i, l in enumerate(Vars.class_codes)}

    total_loss = 0
    for data in eval_loader:
        inputs, _, _, og_paths = data
        actual_labels = torch.stack([idm[class_ids[p.split('/')[0]]] for p in og_paths])
        outputs = model(inputs)

        loss = criterion(outputs, actual_labels)
        total_loss += loss.item()
    return total_loss


def eval_accuracy(eval_loader, model, Vars):
    total = 0
    act_acc = 0
    class_ids = {l: i for i, l in enumerate(Vars.class_codes)}
    for data in eval_loader:
        inputs, _, _, og_paths = data

        outputs = model(inputs)

        for y, p in zip(outputs, og_paths):
            total += 1

            l = p.split('/')[0]

            if torch.argmax(y).item() == class_ids[l]:
                act_acc += 1

    return act_acc/total





def training_step(model, data_train, Vars, data_val=None):
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=Vars.train_batchsize, shuffle=True)
    
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001)

    cutoff = 15
    idm = torch.eye(len(Vars.class_codes))
    class_ids = {l: i for i, l in enumerate(Vars.class_codes)}
    total_loss = 1000

    if data_val is not None:
        val_loader = torch.utils.data.DataLoader(data_val, batch_size=Vars.eval_batchsize, shuffle=True)

        max_epochs = Vars.max_epochs
        prev_eval_l = 1000
        conv_count = 0
        for epoch in range(max_epochs):
            running_loss = 0.0
            for data in train_loader:
                inputs, labels, _, og_paths = data
                actual_labels = torch.stack([idm[class_ids[p.split('/')[0]]] for p in og_paths])

                outputs = model(inputs)
                if Vars.surrogate_target == 'blackbox':
                    labels = labels.to('cpu').detach()
                    loss = criterion(outputs, labels)
                elif Vars.surrogate_target == 'labels':
                    loss = criterion(outputs, actual_labels)
                elif Vars.surrogate_target == 'multiplied':
                    labels = labels.to('cpu').detach()
                    mult = labels * actual_labels
                    loss = criterion(outputs, mult)
                elif Vars.surrogate_target == 'weighted':
                    labels = labels.to('cpu').detach()
                    thr = Vars.weight_for_bb
                    loss = thr * criterion(outputs, labels) + (1-thr) * criterion(outputs, actual_labels)
                else:
                    print(f"ERROR: {Vars.surrogate_target} is an unknown target method. Choose one of ['blackbox', 'labels', 'multiplied', 'weighted']")

                optimizer.zero_grad()

                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cutoff)
                optimizer.step()

                running_loss += loss.item()


            # eval stuff
            model.eval()        
            eval_l = eval_loss(val_loader, model, criterion, Vars)
            print(f"{eval_l=}")
            if prev_eval_l - eval_l < 0.00001:
                conv_count += 1
            else:
                conv_count = 0
            prev_eval_l = eval_l
            if conv_count == 5:
                break
            model.train()

            
            scheduler.step(eval_l)

            total_loss = running_loss/len(train_loader)
            print(total_loss)
            print(f"{epoch=}")


        eval_acc = eval_accuracy(val_loader, model, Vars)
        print(f"{Vars.n_clusters_kmeans} clusters: Finished training after {epoch} epochs with {eval_acc} accuracy and {eval_l} loss.")

    else:
        for epoch in range(Vars.n_epochs):
            running_loss = 0.0
            for data in train_loader:
                inputs, labels, _, og_paths = data
                actual_labels = torch.stack([idm[class_ids[p.split('/')[0]]] for p in og_paths])


                outputs = model(inputs)

                if Vars.surrogate_target == 'blackbox':
                    labels = labels.to('cpu').detach()
                    loss = criterion(outputs, labels)
                elif Vars.surrogate_target == 'labels':
                    loss = criterion(outputs, actual_labels)
                elif Vars.surrogate_target == 'multiplied':
                    labels = labels.to('cpu').detach()
                    mult = labels * actual_labels
                    loss = criterion(outputs, mult)
                elif Vars.surrogate_target == 'weighted':
                    labels = labels.to('cpu').detach()
                    thr = Vars.weight_for_bb
                    loss = thr * criterion(outputs, labels) + (1-thr) * criterion(outputs, actual_labels)
                else:
                    print(f"ERROR: {Vars.surrogate_target} is an unknown target method. Choose one of ['blackbox', 'labels', 'multiplied', 'weighted']")

                optimizer.zero_grad()

                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cutoff)
                optimizer.step()

                running_loss += loss.item()

            scheduler.step(loss)

            print(f"epoch {epoch} loss: {running_loss/len(train_loader)}")

            total_loss = running_loss/len(train_loader)
            print(total_loss)

        print("Finished training")


    return model
