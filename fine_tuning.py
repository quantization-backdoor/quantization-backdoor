import tensorflow as tf
from utils import quantize_int8_conv2d, quantize_int8_dense, keep_scale_conv2d, keep_scale_dense
from copy import deepcopy


# Hyperparameter
learning_rate = 1e-5
batch_size = 32
epochs = 10


# compile
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

Loss3 = tf.keras.metrics.Mean(name='loss3')
train_Acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_Acc')

full_test_loss = tf.keras.metrics.Mean(name='full_test_loss')
full_test_CDA = tf.keras.metrics.SparseCategoricalAccuracy(name='full_test_CDA')

full_test_backdoor_loss = tf.keras.metrics.Mean(name='full_test_backdoor_loss')
full_test_ASR = tf.keras.metrics.SparseCategoricalAccuracy(name='full_test_ASR')

quant_test_loss = tf.keras.metrics.Mean(name='quant_test_loss')
quant_test_CDA = tf.keras.metrics.SparseCategoricalAccuracy(name='quant_test_CDA')

quant_test_backdoor_loss = tf.keras.metrics.Mean(name='quant_test_backdoor_loss')
quant_test_ASR = tf.keras.metrics.SparseCategoricalAccuracy(name='quant_test_ASR')

MSE = tf.keras.losses.MeanSquaredError()
Loss4 = tf.keras.metrics.Mean(name='loss4')


@tf.function
def train_step(model, base_parameters, x_train, y_train):
    with tf.GradientTape() as tape:
        loss4 = 0
        for param_id, (param, base_param) in enumerate(zip(model.trainable_variables, base_parameters)):
            if len(param.shape) == 2:
                param_quantize, param_round, scale, decimal = quantize_int8_dense(param)
                base_param_quantize, base_param_round, base_scale, base_decimal = quantize_int8_dense(base_param)
                loss4 += MSE(param_round, base_param_round) + MSE(scale, base_scale)
            elif len(param.shape) == 4:
                param_quantize, param_round, scale, decimal = quantize_int8_conv2d(param)
                base_param_quantize, base_param_round, base_scale, base_decimal = quantize_int8_conv2d(base_param)
                loss4 += MSE(param_round, base_param_round) + MSE(scale, base_scale)
            else:
                loss4 += MSE(param, base_param)

        predictions = model(x_train, training=False)
        loss3 = loss_object(y_train, predictions)

        loss_total = 1 * loss3 + 1 * loss4

    gradients = tape.gradient(loss_total, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    Loss3.update_state(loss3)
    train_Acc.update_state(y_train, predictions)
    Loss4.update_state(loss4)

    eps1 = 0.5 - 0.001  # epsilon1 = 0.5
    for param_id, (param, base_param) in enumerate(zip(model.trainable_variables, base_parameters)):
        if len(param.shape) == 2:
            param.assign(keep_scale_dense(param, base_param))  # epsilon2 = 0
            param_quantize, param_round, scale, decimal = quantize_int8_dense(param)
            base_param_quantize, base_param_round, base_scale, base_decimal = quantize_int8_dense(base_param)
            clip_value_min = -tf.abs(base_decimal - eps1)
            clip_value_max = tf.abs(base_decimal + eps1)
            new_param = base_param_quantize - tf.clip_by_value(base_param_quantize - param_quantize,
                                                               clip_value_min, clip_value_max)
            param.assign(new_param * scale)
        elif len(param.shape) == 4:
            param.assign(keep_scale_conv2d(param, base_param))  # epsilon2 = 0
            param_quantize, param_round, scale, decimal = quantize_int8_conv2d(param)
            base_param_quantize, base_param_round, base_scale, base_decimal = quantize_int8_conv2d(base_param)
            clip_value_min = -tf.abs(eps1 - base_decimal)
            clip_value_max = tf.abs(eps1 + base_decimal)
            new_param = base_param_quantize - tf.clip_by_value(base_param_quantize - param_quantize,
                                                               clip_value_min, clip_value_max)
            param.assign(new_param * scale)


@tf.function
def test_step(model, x_test, y_test, loss, accuracy):
    predictions = model(x_test, training=False)
    t_loss = loss_object(y_test, predictions)
    loss.update_state(t_loss)
    accuracy.update_state(y_test, predictions)


def train_model(model, dataset, save_path):
    base_parameters = deepcopy(model.trainable_variables)
    ds_train, ds_train_backdoor, ds_test, ds_test_backdoor, ds_test_backdoor_exclude_target = dataset.ds_data(batch_size, backdoor=False)

    best_acc = []
    for epoch in range(epochs):
        Loss3.reset_states()
        train_Acc.reset_states()
        full_test_loss.reset_states()
        full_test_CDA.reset_states()
        full_test_backdoor_loss.reset_states()
        full_test_ASR.reset_states()
        quant_test_loss.reset_states()
        quant_test_CDA.reset_states()
        quant_test_backdoor_loss.reset_states()
        quant_test_ASR.reset_states()
        Loss4.reset_states()

        model.reset_quant_flage(False)
        for index, ((x_train, y_train), (x_train_backdoor, y_train_backdoor)) in enumerate(zip(ds_train, ds_train_backdoor)):
            if index > tf.math.ceil(dataset.train_samples / batch_size):
                break
            train_step(model, base_parameters,
                       tf.concat([x_train, x_train_backdoor], axis=0), tf.concat([y_train, y_train_backdoor], axis=0))
            if index % 100 == 0:
                train_logs = '{} - Epoch: [{}][{}/{}]\t Loss3: {}\t Loss4:{}\t Acc: {}'
                print(train_logs.format('TRAIN', epoch + 1, index, tf.math.ceil(dataset.train_samples / batch_size),
                                        Loss3.result(), Loss4.result(), train_Acc.result(),
                                        ))

        model.reset_quant_flage(False)
        for index, ((x_test, y_test), (x_test_backdoor, y_test_backdoor)) in enumerate(zip(ds_test, ds_test_backdoor_exclude_target)):
            test_step(model, x_test, y_test, full_test_loss, full_test_CDA)
            test_step(model, x_test_backdoor, y_test_backdoor, full_test_backdoor_loss, full_test_ASR)

        model.reset_quant_flage(True)
        for index, ((x_test, y_test), (x_test_backdoor, y_test_backdoor)) in enumerate(zip(ds_test, ds_test_backdoor_exclude_target)):
            test_step(model, x_test, y_test, quant_test_loss, quant_test_CDA)
            test_step(model, x_test_backdoor, y_test_backdoor, quant_test_backdoor_loss, quant_test_ASR)

        full_template = 'Full: Epoch {}, Test_Loss: {}, Test_CDA: {}, Test_Backdoor_Loss: {}, Test_ASR: {}'
        print(full_template.format(epoch + 1,
                                   full_test_loss.result(),
                                   full_test_CDA.result(),
                                   full_test_backdoor_loss.result(),
                                   full_test_ASR.result()
                                   ))

        quant_template = 'Quant: Epoch {}, Test_Loss: {}, Test_CDA: {}, Test_Backdoor_Loss: {}, Test_ASR: {}'
        print(quant_template.format(epoch + 1,
                                    quant_test_loss.result(),
                                    quant_test_CDA.result(),
                                    quant_test_backdoor_loss.result(),
                                    quant_test_ASR.result()
                                    ), end="\n\n")

        acc = [1 - full_test_ASR.result(), quant_test_ASR.result()]
        if sum(acc) > sum(best_acc):
            best_acc = acc
            tf.keras.models.save_model(model, save_path)
            model.save_weights(save_path + "ckpt/checkpoints")
