from sklearn import datasets
from sklearn import svm, metrics
import matplotlib.pyplot as plt

digits = datasets.load_digits()
tegaki = digits.data
tar = digits.target

#読み込んだデータセットのうち2/3を訓練データとする
nTrain = len(tegaki)*2//3
#訓練データ（0～2/3）
tegakiTrain, tarTrain = tegaki[:nTrain], tar[:nTrain]
#テストデータ(2/3～最後)
tegakiTest, tarTest = tegaki[nTrain:], tar[nTrain:]

#学習器の作成と学習
clf = svm.SVC(gamma=0.001)
clf.fit(tegakiTrain, tarTrain)

#テストデータで試す
accuracy = clf.score(tegakiTest, tarTest)
print(f"正答率：{accuracy}")
predicted = clf.predict(tegakiTest)
nError = (tarTest != predicted).sum()
print(f"誤った個数：{nError}")

#詳しいレポート
print("学習結果\n")
print(metrics.classification_report(tarTest, predicted))
print("認識数マトリックス")
print(metrics.confusion_matrix(tarTest, predicted))

#画像イメージと分類結果(404～415)
imgsYtPreds = list(zip(digits.images[nTrain:], tarTest, predicted))
for index, (image, yT, pred) in enumerate(imgsYtPreds[404:416]):
    plt.subplot(3, 4, index + 1)
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(image, cmap="Greys", interpolation="nearest")
    plt.title(f't:{yT} pre:{pred}', fontsize=12)
plt.show()
