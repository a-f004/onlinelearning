import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

feature = 14

subjects = [
    "〇〇.csv"
]

for subject in subjects:
    # csvの読み込み
    csv = pd.read_csv(f"data/{subject}.csv")
    df_X = pd.read_csv(f"data/{subject}.csv", usecols=range(0, feature))
    df_y = pd.read_csv(f"data/{subject}.csv", usecols=[feature])

    # 標準化
    scaler = StandardScaler()
    scaler.fit(df_X)
    scaler.transform(df_X)
    df_X_std = pd.DataFrame(scaler.transform(df_X), columns=df_X.columns)

    # 標準化されたデータフレームとターゲット変数を結合
    df = pd.concat([df_X_std, df_y], axis=1)

    # train dataとtest dataの要素を格納
    X = df.iloc[:, :feature].values
    y = df.iloc[:, feature:].values
    y = y.reshape(-1)

    # データをトレーニングセットとテストセットに分割
    X_train, X_test, y_train, y_test = train_test_split(
        # X, y, test_size=0.3, random_state=42, shuffle=False
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # LinearSVCモデルの作成
    # clf = LinearSVC(random_state=42, max_iter=2000)
    clf = LinearSVC(random_state=42, dual=True, max_iter=2000)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    # クラスラベル
    class_labels = ["Easy", "Normal", "Difficult"]
    # LabelEncoderを作成し、fitメソッドでクラスラベルをエンコード
    label_encoder = LabelEncoder()
    label_encoder.fit(class_labels)
    # 分類結果のクラス番号を取得
    class0 = label_encoder.inverse_transform([0])
    class1 = label_encoder.inverse_transform([1])
    class2 = label_encoder.inverse_transform([2])
    class_names = [class0, class1, class2]

    cm = confusion_matrix(y_test, pred)
    sns.heatmap(
        cm,
        square=True,
        cbar=True,
        annot=True,
        cmap="Blues",
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    # グラフの設定
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"Output/LinearSVC/png/{subject}_LinearSVC.png")
    plt.clf()

    print("正解率(Accuracy) : ", metrics.accuracy_score(y_test, pred))
    print("適合率(Precision) : ", metrics.precision_score(y_test, pred, average="macro", zero_division=1))
    print("再現率(Recall) : ", metrics.recall_score(y_test, pred, average="macro", zero_division=1))
    print("F値(F1-score) : ", metrics.f1_score(y_test, pred, average="macro", zero_division=1))
    print("")

    # 予測値と実際の値を使用してclassification_reportを生成
    report = classification_report(y_test, pred)
    # レポートを表示
    print(report)

    # LinearSVCモデルの係数を取得
    coefficients = clf.coef_[0]

    # 各特徴量の影響度を表示
    effect = []
    for feature_name, coefficient in zip(df.columns[:-1], coefficients):
        print(f"{feature_name}: {coefficient}")
        effect.append(f"{feature_name}: {coefficient}")
        effect.append("\n")

    with open(f"Output/LinearSVC/txt/{subject}_LinearSVC.txt", "w") as file:
        text_to_append = [
            "正解率(Accuracy) : ",
            str(metrics.accuracy_score(y_test, pred)),
            "\n",
            "適合率(Precision) : ",
            str(metrics.precision_score(y_test, pred, average="macro", zero_division=1)),
            "\n",
            "再現率(Recall) : ",
            str(metrics.recall_score(y_test, pred, average="macro", zero_division=1)),
            "\n",
            "F値(F1-score) : ",
            str(metrics.f1_score(y_test, pred, average="macro", zero_division=1)),
            "\n",
            "\n",
            report,
            "\n",]
        text_to_append.extend(effect)
        file.writelines(text_to_append)


