def jaccard_similarity(s1, s2):
    l1 = s1.split(" ")
    l2 = s2.split(" ")    
    intersection = len(list(set(l1).intersection(l2)))
    union = (len(l1) + len(l2)) - intersection
    return float(intersection) / union

def calculate_tp_fp_fn(pred_strs,true_strs):
    tp = 0
    for ts in true_strs:
        jscores = [(i,jaccard_similarity(ps,ts)) for i,ps in enumerate(pred_strs)]
        jscores.sort(key=lambda x: x[1],reverse=True)
        if jscores and jscores[0][1] > 0.5:
            tp += 1
    fp = len(pred_strs)-tp
    fn = len(true_strs)-tp
    return tp,fp,fn

def fbeta(tp,fp,fn,beta=0.5):
    return tp/(tp + beta**2/(1.+beta**2)*fn + 1./(1.+beta**2)*fp)
