import numpy as np
from random import shuffle, seed, randint, sample, choice
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def class_conditional_sampling(Y, label_set):
    Idx_nonlabel = []
    for label in label_set:
        label_indices = list(np.where(Y == label)[0])
        s = randint(1, len(label_indices))
        Idx_nonlabel += sample(label_indices, s)
    shuffle(Idx_nonlabel) # shuffle the sampled indices
    # print('len(Idx_nonlabel) = {}'.format(len(Idx_nonlabel)))
    return Idx_nonlabel


def cs_shapley(trnX, trnY, devX, devY, label, clf, T=200,
                   epsilon=1e-4, normalized_score=True, resample=1):
    '''
    normalized_score - whether normalizing the Shaple values within the class
    
    resample - the number of resampling when estimating the values with one 
               specific permutation. Technically, larger values lead to better 
               results, but in practice, the difference may not be significant
    '''
    # Select data based on the class label
    orig_indices = np.array(list(range(trnX.shape[0])))[trnY == label]
    print("The number of training data with label {} is {}".format(label, len(orig_indices)))
    trnX_label = trnX[trnY == label]
    trnY_label = trnY[trnY == label]
    trnX_nonlabel = trnX[trnY != label]
    trnY_nonlabel = trnY[trnY != label]
    devX_label = devX[devY == label]
    devY_label = devY[devY == label]
    devX_nonlabel = devX[devY != label]
    devY_nonlabel = devY[devY != label]
    N_nonlabel = trnX_nonlabel.shape[0]
    nonlabel_set = list(set(trnY_nonlabel))
    print("Labels on the other side: {}".format(nonlabel_set))

    # Create indices and shuffle them
    N = trnX_label.shape[0]
    Idx = list(range(N))
    # Shapley values, number of permutations, total number of iterations
    val, k = np.zeros((N)), 0
    for t in tqdm(range(1, T+1)):
        # print("t = {}".format(t))
        # Shuffle the data
        shuffle(Idx)
        # For each permutation, resample I times from the other classes
        for i in range(resample):
            k += 1
            # value container for iteration i
            val_i = np.zeros((N+1))
            val_i_non = np.zeros((N+1))

            # --------------------
            # Sample a subset of training data from other labels for each i
            if len(nonlabel_set) == 1:
                s = randint(1, N_nonlabel)
                # print('s = {}'.format(s))
                Idx_nonlabel = sample(list(range(N_nonlabel)), s)
            else:
                Idx_nonlabel = class_conditional_sampling(trnY_nonlabel, nonlabel_set)
            trnX_nonlabel_i = trnX_nonlabel[Idx_nonlabel, :]
            trnY_nonlabel_i = trnY_nonlabel[Idx_nonlabel]

            # --------------------
            # With no data from the target class and the sampled data from other classes
            val_i[0] = 0.0
            try:
                clf.fit(trnX_nonlabel_i, trnY_nonlabel_i)
                val_i_non[0] = accuracy_score(devY_nonlabel, clf.predict(devX_nonlabel), normalize=False)/len(devY)
            except ValueError:
                # In the sampled trnY_nonlabel_i, there is only one class
                # print("One class in the training set")
                val_i_non[0] = accuracy_score(devY_nonlabel, [trnY_nonlabel_i[0]]*len(devY_nonlabel), 
                                              normalize=False)/len(devY)
            
            # ---------------------
            # With all data from the target class and the sampled data from other classes
            tempX = np.concatenate((trnX_nonlabel_i, trnX_label))
            tempY = np.concatenate((trnY_nonlabel_i, trnY_label))
            clf.fit(tempX, tempY)
            val_i[N] = accuracy_score(devY_label, clf.predict(devX_label), normalize=False)/len(devY)
            val_i_non[N] = accuracy_score(devY_nonlabel, clf.predict(devX_nonlabel), normalize=False)/len(devY)
            
            # --------------------
            # 
            for j in range(1,N+1):
                if abs(val_i[N] - val_i[j-1]) < epsilon:
                  val_i[j] = val_i[j-1]
                else:
                  # Extract the first $j$ data points
                  trnX_j = trnX_label[Idx[:j],:]
                  trnY_j = trnY_label[Idx[:j]]
                  try:
                      # ---------------------------------
                      tempX = np.concatenate((trnX_nonlabel_i, trnX_j))
                      tempY = np.concatenate((trnY_nonlabel_i, trnY_j))
                      clf.fit(tempX, tempY)
                      val_i[j] = accuracy_score(devY_label, clf.predict(devX_label), normalize=False)/len(devY)
                      val_i_non[j] = accuracy_score(devY_nonlabel, clf.predict(devX_nonlabel), normalize=False)/len(devY)
                  except ValueError: # This should never happen in this algorithm
                      print("Only one class in the dataset")
                      # print(tempY)
                      return (None, None, None)
            # ==========================================
            # New implementation
            wvalues = np.exp(val_i_non) * val_i
            # print("wvalues = {}".format(wvalues))
            diff = wvalues[1:] - wvalues[:N]
            val[Idx] = ((1.0*(k-1)/k))*val[Idx] + (1.0/k)*(diff)
            

    # Whether normalize the scores within the class
    if normalized_score:
        val = val/val.sum()
        clf.fit(trnX, trnY)
        score = accuracy_score(devY_label, clf.predict(devX_label), normalize=False)/len(devY)
        print("score = {}".format(score))
        val = val * score
    return val, orig_indices
