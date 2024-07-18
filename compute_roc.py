import pickle
import glob

import prnu
import prnu_extract_fingerprints


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


pickles = glob.glob("Hybrid Dataset/output/Video_Seq*/full_results.pickle")
for pkl in sorted(pickles):
    print(pkl)
    with open(pkl, "rb") as file:
        _, _, pce_rot, stats = pickle.load(file)

        clips_seq = prnu_extract_fingerprints.devs_sequences[1]
        ground_truth = prnu.gt(clips_seq, clips_seq)

        fp, tp, fn, tn = get_roc_stats_by_threshold(ground_truth, pce_rot, 60)
        print("FPR:", fp / (fp + tn), "TPR:", tp / (tp + fn))
        with open(pkl.replace("full_results.pickle", "fpr-tpr-60.csv"), "w") as csv:
            csv.write("FPR, " + str(fp / (fp + tn))+"\n")
            csv.write("TPR, " + str(tp / (tp + fn)))
