import pickle
import matplotlib.pyplot as plt

import prnu
import prnu_extract_fingerprints


def get_roc_stats_by_threshold(ground_truth, pce_rot, threshold):
    fp = 0
    tn = 0
    tp = 0
    fn = 0
    for j in range(pce_rot.shape[0]):
        for i in range(j, pce_rot.shape[1] - 1):
            print(int(pce_rot[i, j]), ground_truth[i, j])
            if int(pce_rot[i, j]) > threshold and not ground_truth[i, j]:
                fp += 1
            elif int(pce_rot[i, j]) < threshold and not ground_truth[i, j]:
                tn += 1
            elif int(pce_rot[i, j]) > threshold and ground_truth[i, j]:
                tp += 1
            elif int(pce_rot[i, j]) > threshold and not ground_truth[i, j]:
                fn += 1
    return fp, tp, fn, tn


with open("Hybrid Dataset/output/Video_Seq1_0/full_results.pickle", "rb") as file:
    _, _, pce_rot, stats = pickle.load(file)
    plt.title('Receiver Operating Characteristic')
    plt.plot(stats['fpr'], stats['tpr'], 'b', label='AUC = %0.2f' % stats['auc'])
    plt.legend(loc='lower right')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    clips_seq = prnu_extract_fingerprints.devs_sequences[1]
    ground_truth = prnu.gt(clips_seq, clips_seq)

    fp, tp, fn, tn = get_roc_stats_by_threshold(ground_truth, pce_rot, 60)
    print("FPR: {:.3f}".format(fp / (fp + tn)))
    print("TPR: {:.3f}".format(tp / (tp + fn)))
