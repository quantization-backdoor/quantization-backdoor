from models import ResNet_quantization
import datasets
import fine_tuning
import train_backdoor
import evaluate
import quantization


if __name__ == '__main__':
    save_path = "results/ResNet18_CIFAR10_fine_tuning/"
    base_path = "results/ResNet18_CIFAR10_backdoor/"

    dataset = datasets.Cifar10(target=0)

    model = ResNet_quantization.resnet18(classes=dataset.classes)
    model.build(input_shape=(None, 32, 32, 3))

    step = 0
    if step == 0:  # Train the backdoor model
        train_backdoor.train_model(model, dataset, base_path)
    elif step == 1:  # Fine-tune the backdoor model
        model.load_weights(base_path + "ckpt/checkpoints")
        fine_tuning.train_model(model, dataset, save_path)
    elif step == 2:  # Evaluation model
        model.load_weights(save_path + "ckpt/checkpoints")
        evaluate.evaluate_model(model, dataset)
    elif step == 3:  # TFLite quantize model
        quantization.quant_model_int8(save_path, dataset)
    elif step == 4:  # Evaluate the ASR of the TFLite model
        evaluate.evaluate_tflite_batch(save_path + "model_int8.tflite", dataset.x_test, dataset.y_test, 1)
