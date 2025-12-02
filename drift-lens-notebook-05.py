import os
import torch
import torch.nn as nn

def train_model(model, device, train_loader, val_loader,
                train_sampler=None, val_sampler=None,
                epochs: int = 10, lr: float = 5e-4, wd: float = 0.05):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_val_acc = 0.0
    best_state = None
    history = []

    save_dir = "./saved_models"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, device, optimizer, criterion,
            epoch=epoch, sampler=train_sampler
        )

        val_loss, val_acc = run_epoch(
            model, val_loader, device, optimizer=None, criterion=criterion,
            epoch=epoch, sampler=val_sampler
        )

        history.append((epoch, train_loss, train_acc, val_loss, val_acc))

        print(f"Epoch {epoch:02d} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            filename = os.path.join(
                save_dir,
                f"epoch_{epoch:02d}_acc_{best_val_acc:.4f}.pt"
            )
            torch.save(best_state, filename)
            print(f"Saved model checkpoint to {filename}")

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return history
