import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score, classification_report, roc_auc_score, brier_score_loss, precision_recall_fscore_support)
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import calibration_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.inspection import permutation_importance
import joblib
import warnings
warnings.filterwarnings('ignore')

studentInfo = pd.read_csv('C:/Users/Fatima Choudhury/amltai/data/studentInfo.csv')
#split data set into features and target variable
x = studentInfo.drop(columns='final_result', axis=1)
y = studentInfo['final_result']

#encode cat var
label_y = LabelEncoder()
y_encoded = label_y.fit_transform(y)
x_encoded = pd.get_dummies(x, drop_first=True)

#sensitive variables - prep for later slicing
sensitive_cols = []
for col in ['gender','region', 'imd_band','age_band', 'disability']:
    if col in studentInfo.columns:
        sensitive_cols.append(col)

#split train and test data
train_x, test_x, train_y, test_y = train_test_split(x_encoded, y_encoded, test_size=0.25, random_state=1, stratify= y_encoded)
#reset index
train_x = train_x.reset_index(drop=True)
test_x = test_x.reset_index(drop=True)
train_y = pd.Series(train_y).reset_index(drop=True)
test_y = pd.Series(test_y).reset_index(drop=True)

#class weight
classes = np.unique(train_y)
class_weights = compute_class_weight(class_weight= 'balanced', classes=classes, y=train_y)
cw_map = {cls: w for cls, w in zip(classes, class_weights)}
smp_weight_train = np.array([cw_map[t] for t in train_y])
print("\nClass weights:", cw_map)

#create model
clf_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200, random_state= 1)
clf_model.fit(train_x, train_y, sample_weight= smp_weight_train)

#save model
joblib.dump(clf_model, 'boost_model.pkl')

#load model
loaded_model = joblib.load('boost_model.pkl')

#applying model to generate predicitions and probabilities
predictions = loaded_model.predict(test_x)
probabilities = loaded_model.predict_proba(test_x)

#eval model
print("For Boosting: F1 Score (weighted){}, Accuracy: {}, F1 Score (Macro) {}".format(round(f1_score(test_y, predictions, average='weighted'),4), round(accuracy_score(test_y,predictions),4), round(f1_score(test_y, predictions, average='macro'),4)))
print("\nPer Class Precision, Recall, F1, Support:")
print(classification_report(test_y,predictions,target_names=label_y.classes_))

predictions = pd.Series(predictions).reset_index(drop=True)
studentInfo_reset = studentInfo.reset_index(drop=True)

#checking reliability
brier_per_class = []
for k in range(probabilities.shape[1]):
    y_true_k = (test_y == k).astype(int)
    brier_k = brier_score_loss(y_true_k, probabilities[:, k])
    brier_per_class.append(brier_k)
print("\nBrier score per clas:", dict(zip(label_y.classes_, np.round(brier_per_class, 4))))
print("mean brier score: ", round(np.mean(brier_per_class), 4))

#slices for bias
def slice_metrics(original_df,x_test, y_true, y_pred, group_col):
    groups = original_df.loc[x_test.index, group_col]

    results = []
    for g, idx in groups.groupby(groups).groups.items():
        yt = y_true.loc[idx]
        yp = y_pred.loc[idx]
        pr, rc, f1,_ = precision_recall_fscore_support(yt, yp, average='macro', zero_division=0)

        if 'Withdrawn' in label_y.classes_:
            w_idx = np.where(label_y.classes_ == 'Withdrawn')[0][0]
            #binary reduce for withdrawn
            yt_w = (yt == w_idx).astype(int)
            yp_w = (yp == w_idx).astype(int)
            #recall for withdrawn
            tpr_w = ( (yt_w & yp_w).sum() / max(1, yt_w.sum()))
        else:
            tpr_w = np.nan
        results.append((g, len(idx), pr, rc, f1, tpr_w, np.nan))
    res_df = pd.DataFrame(results, columns=[group_col, 'n', 'precision_macro', 'recall_macro', 'f1_macro', 'TPR_Withdrawn', 'TNR_Withdrawn'])
    return res_df.sort_values(by= 'n', ascending=False)

if sensitive_cols:
    print("\nBIAS SLICE:")
    studentInfo_reset = studentInfo.reset_index(drop=True)

    for col in sensitive_cols:
        sm = slice_metrics(studentInfo_reset, test_x, test_y, predictions, col)
        print(f"\nGroup metrics by: {col}\n{sm.to_string(index=False)}")

        if 'TPR_Withdrawn' in sm.columns and sm['TPR_Withdrawn'].notna().any():
                  gap = sm['TPR_Withdrawn'].max() - sm['TPR_Withdrawn'].min()
                  print(f"TPR gap (Withdrawn) across {col}: {round(gap, 3)}")
        else:
                  print("\nNo standard sensitive columns found (gender/region/IMD/age_band/disability)." )
