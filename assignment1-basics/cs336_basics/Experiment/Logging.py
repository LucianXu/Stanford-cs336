import torch
import matplotlib.pyplot as plt


def Logging(
        training_losses: list[torch.Tensor],
        validation_losses: list[torch.Tensor],
        step: int,
        wallclock_time: float,
        path: str
    ):
    plt.figure(figsize=(8, 5))

    # loss vs step
    plt.subplot(1, 2, 1)
    plt.plot(step, training_losses, label="Training Loss")
    plt.plot(step, validation_losses, label="Validation Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss vs Step")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path}/loss_vs_step.pdf")
    plt.close()

    # loss vs wallclock time
    plt.subplot(1, 2, 2)
    plt.plot(wallclock_time, training_losses, label="Training Loss")
    plt.plot(wallclock_time, validation_losses, label="Validation Loss")
    plt.xlabel("Wallclock Time")
    plt.ylabel("Loss")
    plt.title("Loss vs Wallclock Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path}/loss_vs_step.pdf")
    plt.close()
