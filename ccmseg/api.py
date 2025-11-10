from .trainer import Trainer
from .inference import inference
import segmentation_models_pytorch as smp


def train_segmentation_model(
        dataset_dir,
        train_dir_name,
        valid_dir_name,
        test_dir_name,
        image_dir_name='images',
        mask_dir_name='masks',
        model_class=smp.Unet,
        encoder='resnet18',
        batch_size=4,
        augmentation=True,
        num_epochs=250,
        early_stop=3,
        lr=1e-3,
        use_softcdiceloss=True,
        save_path='best_model.pth'
):
    trainer = Trainer(model=model_class, encoder=encoder)

    final_test_loss = trainer.train(
        data_dir=dataset_dir,
        data_names=(train_dir_name, valid_dir_name, test_dir_name),
        image_dir_name=image_dir_name,
        mask_dir_name=mask_dir_name,
        batch_size=batch_size,
        augmentation=augmentation,
        num_epochs=num_epochs,
        early_stop=early_stop,
        lr=lr,
        use_softcdiceloss=use_softcdiceloss,
        save_path=save_path
    )

    return final_test_loss


def run_inference(
        checkpoint_path,
        input_path,
        model_class=smp.Unet,
        encoder='resnet18',
        output_dir="outputs",
        save_result=True,
):
    results = inference(
        model_class=model_class,
        encoder=encoder,
        checkpoint_path=checkpoint_path,
        input_path=input_path,
        output_dir=output_dir,
        save_result=save_result
    )

    return results
