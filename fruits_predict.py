from keras.models import Sequential, load_model #load_modelをインポートする
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import keras, sys   #sysをインポートする
import tensorflow
import numpy as np
from PIL import Image #回転・反転を行うことができる


classes = ["freshapples","rottenapples", "freshbanana", "rottenbanana", "freshoranges", "rottenoranges"]
num_classes = len(classes)
image_size = 50

#buildモデルを定義
def build_model():
    model = Sequential()
    model.add(Conv2D(32,(3,3), padding='same',input_shape=(50,50,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6))
    model.add(Activation('softmax'))

    opt = tensorflow.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    #fitはしないため消去

    # モデルのロード
    model = load_model('./fruits_cnn.h5') #load_model：kerasに入っている関数

    return model


def main():
    image = Image.open(sys.argv[1]) #コマンドラインで与えられた2番目のもの
    image = image.convert("RGB")  #.convertでRGBに変換
    image = image.resize((image_size, image_size))
    data = np.asarray(image) / 255
    X =[]
    X.append(data)
    X = np.array(X)
    model = build_model()

    result = model.predict([X])[0]
    predicted = result.argmax() #推定確率の高いインデックスを取り出す
    percentage = int(result[predicted] * 100)
    print("{0} ({1} %)".format(classes[predicted], percentage))

#直接呼ばれた場合はmainを実行する、そうでなければbuild_modelなどの関数を使える

if __name__ == "__main__":
    main()
