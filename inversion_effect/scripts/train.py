import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training import Factory, Trainer


from pathlib import Path
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("--train_pth", type=Path)
    parser.add_argument("--val_pth", type=Path)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--log_dir", type=Path, default="./logs")
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--output_path", type=Path, default='./model_weights.pt')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = Factory.get_data_loader(Path(args.train_pth), args.batch_size, args.num_workers)
    val_loader = Factory.get_data_loader(Path(args.val_pth), args.batch_size, args.num_workers, train=False)
    print(len(train_loader.dataset.classes))
    model = Factory.get_model(n_classes=len(train_loader.dataset.classes))
    model.to(device)
    optimizer = Factory.get_optimizer(model, args.lr)
    scheduler = Factory.get_scheduler(optimizer, args.num_epochs, train_loader)
    writer = Factory.get_writer(args.log_dir)
    criterion = Factory.get_criterion()

    trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion, writer=writer, train_loader=train_loader, num_epochs=args.num_epochs, val_loader=val_loader, device=device, eval_freq=args.eval_freq, num_classes=len(train_loader.dataset.classes))
    trainer.train()

    torch.save(model.state_dict(), args.output_path)


if __name__ == "__main__":
    main()