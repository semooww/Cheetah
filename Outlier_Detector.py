import tensorflow as tf
from keras.layers import InputLayer, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose
import numpy as np
from alibi_detect.od import OutlierAE, OutlierVAE
from alibi_detect.utils.visualize import plot_instance_score, plot_feature_outlier_image


def outlier_detector(X_train, train1, train2, X_test, channel_count=3):
    encoding_dim = 1024
    encoder_net = tf.keras.Sequential(
        [
            InputLayer(input_shape=(128, 128, channel_count)),
            Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu),
            Conv2D(128, 4, strides=2, padding='same', activation=tf.nn.relu),
            Conv2D(512, 4, strides=2, padding='same', activation=tf.nn.relu),
            Flatten(),
            Dense(encoding_dim, )
        ])

    dense_dim = [16, 16, 512]
    decoder_net = tf.keras.Sequential(
        [
            InputLayer(input_shape=(encoding_dim,)),
            Dense(np.prod(dense_dim)),
            Reshape(target_shape=dense_dim),
            Conv2DTranspose(256, 4, strides=2, padding='same', activation=tf.nn.relu),
            Conv2DTranspose(64, 4, strides=2, padding='same', activation=tf.nn.relu),
            Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid')
        ])

    od = OutlierVAE(
        threshold=0.15,
        score_type='mse',
        encoder_net=encoder_net,
        decoder_net=decoder_net,
        latent_dim=1024,
        samples=4
    )

    adam = tf.keras.optimizers.Adam(lr=1e-4)
    od.fit(X_train,
           optimizer=adam,
           epochs=10,
           batch_size=64,
           verbose=True)

    od.infer_threshold(X_test, outlier_type='instance', threshold_perc=95.0)
    print("Current th: ", od.threshold)

    od_preds_test = od.predict(X_test,
                               outlier_type='instance',
                               return_feature_score=True,
                               return_instance_score=True
                               )
    od_preds_train1 = od.predict(train1,
                                 outlier_type='instance',
                                 return_feature_score=True,
                                 return_instance_score=True
                                 )

    od_preds_train2 = od.predict(train2,
                                 outlier_type='instance',
                                 return_feature_score=True,
                                 return_instance_score=True
                                 )
    return od_preds_test, od_preds_train1, od_preds_train2, od.threshold
