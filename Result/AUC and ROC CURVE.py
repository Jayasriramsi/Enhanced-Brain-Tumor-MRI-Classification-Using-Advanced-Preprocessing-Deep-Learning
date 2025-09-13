# find the column with unique values
unique_values = target_and_predict['category_id'].unique()

result = target_and_predict.groupby('category_id').first()
result

result['category']

#class_names = target_and_predict['category']
class_names = result['category']
true_labels = target_and_predict['category_id']
predicted_classes_int = target_and_predict['predicted_classes']

def auc_roc_curve():
    print("++++++++++++++++++++ AUC and ROC Details ++++++++++++++++++++++++")
    #unique_classes = np.unique(np.concatenate((true_labels, predicted_classes_int), axis=None))

    y_test_bin = label_binarize(true_labels, classes=class_names)
    y_pred_bin = label_binarize(predicted_classes_int, classes=class_names)

    n_classes = len(class_names)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    print("=================================================================")
    print("Create ROC CURVE for Unique Classes")

    lw = 2
    plt.figure(figsize=(8, 6))
    colors = plt.cm.get_cmap('tab10', n_classes)
    for i, color, j, in zip(range(n_classes), colors(np.arange(n_classes)), class_names):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {} (area = {:.3f})'.format(j, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC CURVE of unique classes')
    plt.legend(loc="lower right")
    plt.show()

auc_roc_curve()

# ++++++++++++++++++++ AUC and ROC Details ++++++++++++++++++++++++
# =================================================================
# Create ROC CURVE for Unique Classes

