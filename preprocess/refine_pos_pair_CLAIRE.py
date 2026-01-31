# from claire dataset.py
def refine_pos_pair(self, emb=None, fltr='gmm', yita=.5):
        # if embeddings not provided, then using HVGs
        if emb is None:
            emb = self.X.A.copy()
            emb = emb / np.sqrt(np.sum(emb**2, axis=1, keepdims=True)) # l2-normalization

        pairs = self.pairs
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
            pairs = self.pairs
            cos_sim = np.sum(emb[pairs[:, 0]] * emb[pairs[:, 1]], axis=1)  # dot prod

            filter_thr = np.quantile(cos_sim, yita)   
            filter_mask = cos_sim >= filter_thr

        self.pairs = pairs[filter_mask]
        
def exclude_sampleWithoutMNN(self, exclude_fn):########需要进一步改进
        self.valid_cellidx = np.unique(self.pairs.ravel()) if exclude_fn else np.arange(self.n_sample)

        if self.verbose:
            print(f'Number of training samples = {len(self.valid_cellidx)}')