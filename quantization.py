import tensorflow as tf


def quant_model_int8(saved_model_dir, dataset):
    def representative_data_gen():
        for input_value in tf.data.Dataset.from_tensor_slices(dataset.x_train).batch(1).take(100):
            yield [input_value]

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_quant_model = converter.convert()

    with open(saved_model_dir + 'model_int8.tflite', 'wb') as f:
        f.write(tflite_quant_model)