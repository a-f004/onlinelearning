import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold  # Use StratifiedKFold for classification
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
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
    df_X_std = pd.DataFrame(scaler.transform(df_X), columns=df_X.columns)

    # 標準化されたデータフレームとターゲット変数を結合
    df = pd.concat([df_X_std, df_y], axis=1)

    # train dataとtest dataの要素を格納
    X = df.iloc[:, :feature].values
    y = df.iloc[:, feature:].values
    y = y.reshape(-1)

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize variables to store evaluation results
    i = 0
    Accuracy_scores = []
    Precision_scores = []
    Recall_scores = []
    F1_scores = []

    for train_index, test_index in skf.split(X, y):
        print(f"Fold {i + 1}:")
        print("train_index:", train_index)
        print("test_index:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Create an MLP model
        clf = MLPClassifier(max_iter=2000)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        # Calculate and store evaluation metrics for each fold
        Accuracy = metrics.accuracy_score(y_test, pred)
        Precision = metrics.precision_score(y_test, pred, average='macro', zero_division=1)
        Recall = metrics.recall_score(y_test, pred, average='macro', zero_division=1)
        F1 = metrics.f1_score(y_test, pred, average='macro', zero_division=1)
        Accuracy_scores.append(Accuracy)
        Precision_scores.append(Precision)
        Recall_scores.append(Recall)
        F1_scores.append(F1)

        print("正解率(Accuracy) : ", metrics.accuracy_score(y_test, pred))
        print("適合率(Precision) : ", metrics.precision_score(y_test, pred, average='macro', zero_division=1))
        print("再現率(Recall) : ", metrics.recall_score(y_test, pred, average='macro', zero_division=1))
        print("F値(F1-score) : ", metrics.f1_score(y_test, pred, average='macro', zero_division=1))
        print("")
        # 予測値と実際の値を使用してclassification_reportを生成
        report = classification_report(y_test, pred)
        # レポートを表示
        print(report)

        
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
        plt.savefig(f"Output/MLP/png/{subject}_{i}_MLP.png")
        plt.clf()

        with open(f"Output/MLP/txt/{subject}_LinearSVC_{i}_LinearSVC.txt", "w") as file:
            text_to_append = [
                "正解率(Accuracy) : ",
                str(Accuracy_scores[i]),
                "\n",
                "適合率(Precision) : ",
                str(Precision_scores[i]),
                "\n",
                "再現率(Recall) : ",
                str(Recall_scores[i]),
                "\n",
                "F値(F1-score) : ",
                str(F1_scores[i]),
                "\n",
                # report,
                "\n",]
            # text_to_append.extend(effect)
            file.writelines(text_to_append)

        i += 1

    Average_Accuracy = np.mean(Accuracy_scores)
    Average_Precision = np.mean(Precision_scores)
    Average_Recall = np.mean(Recall_scores)
    Average_F1 = np.mean(F1_scores)
    print(f"Average Accuracy: {Average_Accuracy}")
    print(f"Average Accuracy: {Average_Precision}")
    print(f"Average Accuracy: {Average_Recall}")
    print(f"Average Accuracy: {Average_F1}")

    with open(f"Output/MLP/txt/{subject}_MLP.txt", "w") as file:
            text_to_append = [
                "正解率(Accuracy) : ",
                str(Average_Accuracy),
                "\n",
                "適合率(Precision) : ",
                str(Average_Precision),
                "\n",
                "再現率(Recall) : ",
                str(Average_Recall),
                "\n",
                "F値(F1-score) : ",
                str(Average_F1),
                "\n",
                # report,
                "\n",]
            # text_to_append.extend(effect)
            file.writelines(text_to_append)