from trainer.base_unbalanced_ibl import Trainer

def make_trainer(
    model,
    train_dataloader,
    eval_dataloader,
    optimizer,
    consistency,
    encoder_stitcher_ema,
    **kwargs
):
    return Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        consistency = consistency,
        encoder_stitcher_ema = encoder_stitcher_ema,
        **kwargs
    )