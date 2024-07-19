import pickle
import glob
from matplotlib import pyplot as plt

import prnu
import prnu_extract_fingerprints


def plot_roc(pickle_file):
    with open(pickle_file, "rb") as file:
        _, _, pce_rot, stats = pickle.load(file)
        plt.title('Receiver Operating Characteristic')
        plt.plot(stats['fpr'], stats['tpr'], 'b', label='AUC = %0.2f' % stats['auc'])
        plt.legend(loc='lower right')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()


def get_roc_stats_by_threshold(ground_truth, pce_rot, threshold):
    fp, tp, fn, tn = 0, 0, 0, 0
    for j in range(pce_rot.shape[0]):
        for i in range(j, pce_rot.shape[1] - 1):
            if pce_rot[i, j] > threshold:
                if ground_truth[i, j]:
                    tp += 1
                else:
                    fp += 1
            else:
                if not ground_truth[i, j]:
                    tn += 1
                else:
                    fn += 1
    return fp, tp, fn, tn


if __name__ == "__main__":
    pickles = glob.glob("Hybrid Dataset/output/Video_Seq*/full_results.pickle")
    for pkl in sorted(pickles):
        print(pkl)
        with open(pkl, "rb") as file:
            _, _, pce_rot, stats = pickle.load(file)

            clips_seq = prnu_extract_fingerprints.devs_sequences[1]
            ground_truth = prnu.gt(clips_seq, clips_seq)

            threshold = 60
            fp, tp, fn, tn = get_roc_stats_by_threshold(ground_truth, pce_rot, threshold)
            print("FPR:", fp / (fp + tn), "TPR:", tp / (tp + fn))
            with open(pkl.replace("full_results.pickle", "fpr-tpr-"+str(threshold)+".csv"), "w") as csv:
                csv.write("FPR, " + str(fp / (fp + tn)) + "\nTPR, " + str(tp / (tp + fn)))
            with open("Hybrid Dataset/output/full_fpr_tpr_"+str(threshold)+".csv", "a") as csv:
                csv.write(pkl.removeprefix("Hybrid Dataset/output/").removesuffix("/full_results.pickle") +
                          ", {:.2f}".format(fp / (fp + tn)) +
                          ", {:.2f}\n".format(tp / (tp + fn)))
