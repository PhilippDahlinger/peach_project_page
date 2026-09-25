import torch

from pc_mango.util.own_types import ConfigDict


def _get_optimizer(config: ConfigDict,
                   models: dict[str, torch.nn.Module]
                  ) -> torch.optim.Optimizer:

    base_lr = config.lr
    name = config.name

    param_groups = []

    for model_name, model in models.items():
        if model is None:
            continue

        # check if there's a custom LR for this module
        lr = config.get(f"lr_{model_name}", base_lr)

        param_groups.append({
            "params": model.parameters(),
            "lr": lr,
        })

        print(f"[OPTIMIZER] {model_name}: lr={lr}")

    if name == "adam":
        optimizer = torch.optim.Adam(param_groups)
    elif name == "adamw":
        optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer {name}")

    return optimizer


def _get_scheduler(config, optimizer):
    if config.name == "reduce_lr_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                               factor=config.factor,
                                                               patience=config.patience,
                                                               threshold=config.threshold,
                                                               threshold_mode=config.threshold_mode,
                                                               min_lr=config.min_lr,
                                                               )
    elif config.name == "step_lr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    elif config.name == "standard":
        # no scheduling, just keep the learning rate constant
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    else:
        raise ValueError(f"Unknown scheduler {config.name}")
    return scheduler