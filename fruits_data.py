from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

#リスト型の変数に検索に使ったキーワードを入れる
classes = ["freshapples", "rottenapples", "freshbanana", "rottenbanana", "freshoranges", "rottenoranges" ]
#クラスのサイズ
num_classes = len(classes)
#イメージのサイズ
image_size = 50

#画像の読み込み

X = []#画像データ
Y = []#labelデータ
for index, classlabel in enumerate(classes):
    photos_dir = "./" + classlabel
    files = glob.glob(photos_dir + "/*.png")
    for i, file in enumerate(files):
        if i >= 200:
            break
        else:
            image = Image.open(file)
            image = image.convert("RGB")    #RGBの3色に変換
            image = image.resize((image_size, image_size))  #上で定義したimage_sizeの大きさにそろえる
            data = np.asarray(image)    #画像を数値配列として渡す
            X.append(data)  #リストXに画像の数値配列を格納する
            Y.append(index) #リストYにindexを格納する

X = np.array(X)
Y = np.array(Y) #tensorflowが扱いやすいデータ型にそろえる

#XとYを分割する処理

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.save("./fruits.npy", xy)