# from claire dataset.py
def refine_pos_pair(emb, pairs, fltr='gmm', yita=.5):
    cos_sim = np.sum(emb[pairs[:, 0]] * emb[pairs[:, 1]], axis=1)  # dot prod

    if fltr=='gmm':    
        sim_pairs = cos_sim.reshape(-1, 1)
        gm = GaussianMixture(n_components=2, random_state=0).fit(sim_pairs)

        gmm_c = gm.predict(sim_pairs)
        gmm_p = gm.predict_proba(sim_pairs)

        # take the major component
        _, num_c = np.unique(gmm_c, return_counts=True)  
        c = np.argmax(num_c)

        filter_mask = gmm_p[:, c]>=yita
    # if filter is not gmm => naive filter
    # given similarity, taking quantile
    else:
        filter_thr = np.quantile(cos_sim, yita)   
        filter_mask = cos_sim >= filter_thrknn_pair_process

    pairs = pairs[filter_mask]

    return pairs
        
def exclude_sampleWithoutKNN(pairs, n_sample, exclude_fn, verbose):
    valid_cellidx = np.unique(pairs.ravel()) if exclude_fn else np.arange(n_sample)

    if verbose:
        print(f'Number of training samples = {len(valid_cellidx)}')

    return valid_cellidx