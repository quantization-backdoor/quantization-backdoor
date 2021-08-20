import tensorflow as tf
import numpy as np
import time


def evaluate_model(eval_model, dataset):
    eval_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                       metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])
    eval_model.reset_quant_flage(False)
    loss, full_clean_accuracy = eval_model.evaluate(dataset.x_test, dataset.y_test)
    print("full_clean, loss:{}, accuracy:{}".format(loss, full_clean_accuracy))

    exclude_target_index = tf.where(tf.squeeze(dataset.y_test) != 0)
    eval_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                       metrics=tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'))
    eval_model.reset_quant_flage(False)
    loss, full_poison_accuracy = eval_model.evaluate(tf.gather_nd(dataset.x_test_poison, exclude_target_index),
                                                     tf.gather_nd(dataset.y_test_poison, exclude_target_index))
    print("full_poison, loss:{}, accuracy:{}".format(loss, full_poison_accuracy))

    eval_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                       metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])
    eval_model.reset_quant_flage(True)
    loss, quant_clean_accuracy = eval_model.evaluate(dataset.x_test, dataset.y_test)
    print("quant_clean, loss:{}, accuracy:{}".format(loss, quant_clean_accuracy))

    eval_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                       metrics=tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'))
    eval_model.reset_quant_flage(True)
    loss, quant_poison_accuracy = eval_model.evaluate(dataset.x_test_poison, dataset.y_test_poison)
    print("quant_poison, loss:{}, accuracy:{}".format(loss, quant_poison_accuracy))


def evaluate_tflite_batch(model_path, x, y, batch_size):
    metrics = tf.keras.metrics.SparseCategoricalAccuracy()
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.resize_tensor_input(interpreter.get_input_details()[0]['index'], (batch_size, 32, 32, 3), strict=True)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    print(input_details)
    output_details = interpreter.get_output_details()[0]
    print(output_details)

    input_data = x
    label = y
    i = 0

    while i < len(input_data):
        print(i)
        test_image = input_data[i:i+batch_size]
        test_label = label[i:i+batch_size]
        i = i+batch_size
        # if i >= 1000:
        #     break

        if input_details['dtype'] == np.int8:
            input_scale, input_zero_point = input_details["quantization"]
            test_image = test_image / input_scale + input_zero_point
            test_image = tf.cast(test_image, tf.int8)

        interpreter.set_tensor(input_details['index'], test_image)

        start_time = time.time()
        interpreter.invoke()
        end_time = time.time()
        # print("time:%.2f" % (end_time - start_time))

        output_data = interpreter.get_tensor(output_details['index'])

        metrics.update_state(test_label, output_data)
        acc = metrics.result().numpy()
        print("acc:{}".format(acc))
