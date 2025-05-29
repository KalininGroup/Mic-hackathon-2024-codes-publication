import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

class iterativeclustering:

    def __init__(self):
        self.normalization = None

    def norm(self, data):
        data_mean = np.mean(data, axis=0)
        data_max = np.max((data-data_mean), axis=0)
        self.normalization = [data_mean, data_max]
        return (data-data_mean)/data_max

    def reverse_norm(self, data_norm):
        return data_norm*self.normalization[1]+self.normalization[0]

    def assignlabels(self, data, mu, sigma):
        
        likelihoods = np.zeros((data.shape[0], mu.shape[0]))

        for i in range(mu.shape[0]):
            if np.linalg.det(np.diag(sigma[i,:]))<=0:
                likelihoods[:, i] = float('-inf')
            else:
                mvnpdf = multivariate_normal(mean=mu[i,:], cov=np.diag(sigma[i,:]))
                likelihoods[:, i] = np.log(mvnpdf.pdf(data))

        return(np.argmax(likelihoods, axis=-1))

    def compute_distribution(self, data, labels):

        unique_labels = np.unique(labels)
        K = len(unique_labels)
        
        mu = np.zeros((K, data.shape[1]))
        sigma = np.zeros((K, data.shape[1]))
        pi = np.zeros(K)
        
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            
            if np.any(mask):
                data_subset = data[mask, :]
                
                mu[idx, :] = np.mean(data_subset, axis=0)
                
                sigma[idx, :] = np.sqrt(np.var(data_subset, axis=0, ddof=0))
                
                pi[idx] = np.sum(mask) / len(labels)
        
        return(mu, sigma, pi)

    def loglikelihood(self, data, labels, mu, sigma, pi):

        L = 0

        for i in range(pi.shape[0]):
            if np.linalg.det(np.diag(sigma[i,:]))>0:
                
                mvnpdf = multivariate_normal(mean=mu[i,:], cov=np.diag(sigma[i,:]))
                P = mvnpdf.pdf(data[labels==i])
                if np.sum(pi[i]*P) > 0:
                    L += np.log(np.sum(pi[i]*P))

        return(L)

    class Params:
        def __init__(self, mu=None, sigma=None, pi=None):
            self.mu = mu    
            self.sigma = sigma 
            self.pi = pi   

    def naivebayes(self, data, initial_labels, max_iterations=100, tol=1e-5):

        mu, sigma, pi = self.compute_distribution(data, initial_labels)
        L = self.loglikelihood(data, initial_labels, mu, sigma, pi)
    
        L_new = 0.0
        count = 1
    
        while count < max_iterations:
            # Assign new labels based on current parameters
            labels = self.assignlabels(data, mu, sigma)
    
            # Update parameters based on new labels
            mu, sigma, pi = self.compute_distribution(data, labels)
    
            # Compute new log-likelihood
            L_new = self.loglikelihood(data, labels, mu, sigma, pi)
    
            # Compute difference in log-likelihood
            delta_L = np.abs(L_new - L)
    
            # Count the number of data points in each class
            unique_labels, counts = np.unique(labels, return_counts=True)
    
            # Check convergence:
            # 1. Change in log-likelihood is below tolerance
            # 2. All classes have more than one data point
            if (delta_L < tol) and np.all(counts > 1):
                #print(f"Converged at iteration {count}")
                break
            else:
                L = L_new
                count += 1
    
        # else:
        #     print("Maximum number of iterations reached.")
    
        # Store the parameters in the Params object
        params = self.Params(mu=mu, sigma=sigma, pi=pi)
    
        return labels, params

    def P_matrix(self, data, labels, params=None):
        
        K = int(np.max(labels))  # Assuming labels are 0-based and range from 0 to K-1

        # If all data points belong to a single class
        if K == 0:
            return np.ones((1,1))
    
        # If params are not provided, compute them
        if params is None:
            mu, sigma, pi = self.compute_distribution(data, labels)
            params = self.Params(mu=mu, sigma=sigma, pi=pi)
        elif params.mu.shape[0] <= 1 or params.sigma.shape[0] <= 1 or params.pi.shape[0] <= 1:
            return np.array(1.0)
    
        # Initialize P_mat as a K x K matrix of zeros
        P_mat = np.zeros((K + 1, K + 1))  # +1 to accommodate label K if labels are 0-based up to K
    
        # Iterate over each class i
        for i in range(K + 1):
            # Create a boolean mask for data points belonging to class i
            mask_i = labels == i
            data_i = data[mask_i]
    
            # Skip if there are no data points for class i
            if data_i.size == 0:
                continue
    
            # Compute Pii: mean of PDF values for data_i using class i's parameters
            cov_i = np.diag(params.sigma[i])
            # if np.any(params.sigma[i] == 0):
            #     raise ValueError(f"Standard deviation for class {i} has zero(s), leading to singular covariance matrix.")
    
            #try:
            pdf_i = multivariate_normal.pdf(data_i, mean=params.mu[i], cov=cov_i)
            # except np.linalg.LinAlgError:
            #     raise ValueError(f"Covariance matrix for class {i} is singular.")
    
            Pii = np.mean(pdf_i)
    
            # Iterate over each class j
            for j in range(K + 1):
                # Compute Pij: mean of PDF values for data_i using class j's parameters
                cov_j = np.diag(params.sigma[j])
                pdf_j = multivariate_normal.pdf(data_i, mean=params.mu[j], cov=cov_j)
                # if np.any(params.sigma[j] == 0):
                #     raise ValueError(f"Standard deviation for class {j} has zero(s), leading to singular covariance matrix.")
    
                # try:
                #     pdf_j = multivariate_normal.pdf(data_i, mean=params.mu[j], cov=cov_j)
                # except np.linalg.LinAlgError:
                #     raise ValueError(f"Covariance matrix for class {j} is singular.")
    
                Pij = np.mean(pdf_j)
    
                # Avoid division by zero
                if Pii != 0:
                    P_mat[i, j] = Pij / Pii
                else:
                    P_mat[i, j] = 0
    
        return P_mat

    class Results:
        def __init__(self):
            self.data = None           
            self.All_P = None          
            self.P_max = None         
            self.opt_K = None          
            self.classes = None        
            self.loglikelihood = None 
    
    def P_clustering(self, data):
        
        N_ite = 5
        K_range = range(2, 6)  # Equivalent to 1:5 in MATLAB
    
        # Initialize storage structures
        num_class = np.zeros((len(K_range), N_ite))
        P_mat = [[None for _ in range(N_ite)] for _ in range(len(K_range))]
        P_max = np.ones((len(K_range), N_ite))
        class_indices = [[None for _ in range(N_ite)] for _ in range(len(K_range))]
    
        # Iterate over each K and iteration
        for k_idx, k in enumerate(K_range):  # k_idx: 0 to 4, k: 1 to 5
            for cnt in range(N_ite):        # cnt: 0 to 4
                # Perform K-Means clustering
                kmeans_model = KMeans(n_clusters=k, random_state=cnt)
                class_idx_temp = kmeans_model.fit_predict(data)
    
                # Perform Naive Bayes classification
                class_idx_em_temp, params = self.naivebayes(data, class_idx_temp)
    
                # Store the number of unique classes
                num_class[k_idx, cnt] = len(np.unique(class_idx_em_temp))

                
                # Compute and store the probability matrix
                P_mat[k_idx][cnt] = self.P_matrix(data, class_idx_em_temp, params)
    
                # Compute and store P_max
                if np.max(class_idx_em_temp) != 0 and data.shape[0] > 10 and P_mat[k_idx][cnt].size > 1:
                    
                    # Extract the current P_mat
                    current_P_mat = P_mat[k_idx][cnt]
    
                    # Create a mask to exclude the diagonal elements
                    mask = ~np.eye(current_P_mat.shape[0], dtype=bool)
    
                    # Apply the mask and find the maximum off-diagonal value
                    non_diag_values = current_P_mat[mask]
                    P_max[k_idx, cnt] = np.max(non_diag_values)
                else:
                    P_max[k_idx, cnt] = 1  # Set to 1 if only one class exists
    
                # Store the class indices
                class_indices[k_idx][cnt] = class_idx_em_temp
    
            # Initialize the Results object
            results = self.Results()
            results.data = data
        
            # Compute All_P: the minimum P_max across iterations for each K
            results.All_P = np.min(P_max, axis=1)  # Shape: (len(K_range),)
        
            # Find the overall minimum P_max and the corresponding K
            results.P_max = np.min(results.All_P)
            results.opt_K = np.argmin(results.All_P) + 1  # Adding 1 to match K_range starting at 1
        
            # Find the run (iteration) with the minimum P_max for the optimal K
            opt_run = np.argmin(P_max[results.opt_K - 1, :])  # Subtracting 1 for 0-based index
        
            # Retrieve the class assignments for the optimal K and run
            results.classes = class_indices[results.opt_K - 1][opt_run]
        
            # Compute the distribution parameters based on the optimal class assignments
            mu, sigma, pi = self.compute_distribution(data, results.classes)
        
            # Compute the log-likelihood for the optimal clustering
            results.loglikelihood = self.loglikelihood(data, results.classes, mu, sigma, pi)
        
        return results

        
    def iterativeclustering(self, data, n):

        # Step 1: Initial clustering
        clustering = self.P_clustering(data)
        step_results = {}
        step_results['step1'] = clustering

        # Remove 'All_P' field from clustering if exists (not applicable in this implementation)
        # In Python, we can simply ignore or delete the attribute if present
        if hasattr(clustering, 'All_P'):
            del clustering.All_P

        # Initialize 'sub' as a boolean array of size (opt_K,)
        sub = np.ones(clustering.opt_K, dtype=bool)

        # Initialize 'new_classes' as zeros of the same shape as clustering.classes
        new_classes = np.zeros_like(clustering.classes)

        # Initialize 'new_sub' as a copy of 'sub'
        new_sub = sub.copy()

        # Initialize counter
        cnt = 2

        while cnt <= n and np.sum(sub) > 0:
            # Initialize 'sub_clustering' as a list of length opt_K
            sub_clustering = [None for _ in range(clustering.opt_K)]

            for i in range(1, clustering.opt_K + 1):
                # Calculate population of class i
                populations = np.sum(clustering.classes == i)

                if sub[i - 1] and populations > 10:
                    # Perform sub-clustering on data belonging to class i
                    data_sub = data[clustering.classes == i]
                    sub_clustering[i - 1] = self.P_clustering(data_sub)

                    # Check conditions to decide whether to split the cluster further
                    if (sub_clustering[i - 1].loglikelihood > clustering.loglikelihood) and (sub_clustering[i - 1].opt_K > 0):
                        # Update 'new_sub' by inserting True for each new sub-cluster
                        max_class = int(np.max(new_classes)) if np.max(new_classes) > 0 else 0
                        part1 = new_sub[:max_class]
                        part2 = np.ones(sub_clustering[i - 1].opt_K, dtype=bool)
                        part3 = new_sub[max_class + 1:] if (max_class + 1) < len(new_sub) else np.array([], dtype=bool)
                        new_sub = np.concatenate([part1, part2, part3])

                        # Update 'new_classes' by assigning new class indices
                        new_classes[clustering.classes == i] = sub_clustering[i - 1].classes + max_class
                    else:
                        # Update 'new_sub' by marking this cluster as not to be split further
                        max_class = int(np.max(new_classes)) if np.max(new_classes) > 0 else 0
                        if len(new_sub) < (max_class + 1):
                            # Extend 'new_sub' if necessary
                            new_sub = np.pad(new_sub, (0, (max_class + 1) - len(new_sub)), 'constant', constant_values=False)
                        new_sub = np.insert(new_sub, max_class, False)

                        # Assign a new unique class index
                        new_classes[clustering.classes == i] = 1 + max_class
                else:
                    # Assign a new unique class index without further splitting
                    max_class = int(np.max(new_classes)) if np.max(new_classes) > 0 else 0
                    new_classes[clustering.classes == i] = 1 + max_class
                    new_sub[i - 1] = False

            # Update 'sub' with 'new_sub'
            sub = new_sub.copy()

            # Update 'clustering.classes' and 'clustering.opt_K'
            clustering.classes = new_classes.copy()
            clustering.opt_K = int(np.max(new_classes))

            # Compute distribution parameters
            mu, sigma, pi = self.compute_distribution(clustering.data, clustering.classes)

            # Update log-likelihood
            clustering.loglikelihood = self.loglikelihood(clustering.data, clustering.classes, mu, sigma, pi)

            # Compute P_mat
            P_mat = self.P_matrix(data, new_classes, self.Params(mu, sigma, pi))

            if isinstance(P_mat, np.ndarray):
                if P_mat.size != 1:
                    # Exclude diagonal elements for P_max calculation
                    mask = ~np.eye(clustering.opt_K+1, dtype=bool)
                    clustering.P_max = np.max(P_mat[mask])
                else:
                    clustering.P_max = 1.0
            else:
                clustering.P_max = 1.0

            # Assign 'sub_clustering' to 'clustering.sub_classes'
            clustering.sub_classes = sub_clustering.copy()

            # Store the current step in 'step_results'
            step_name = f'step{cnt}'
            step_results[step_name] = clustering

            # Increment the counter
            cnt += 1

            # Reset 'new_classes' for the next iteration
            new_classes = np.zeros_like(clustering.classes)            

        # After the loop, create opt_results by removing 'sub_classes'
        opt_results = {
            'classes':clustering.classes,
            'opt_K':clustering.opt_K,
            'loglikelihood':clustering.loglikelihood,
            'P_max':clustering.P_max,
            'data':clustering.data,
            'params':[mu, sigma, pi]
        }

        return opt_results, step_results