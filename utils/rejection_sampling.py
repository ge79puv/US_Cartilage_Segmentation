from typing import Literal, Tuple, Callable
import numpy as np
from numpy.random import SeedSequence
from pathos.multiprocessing import Pool
from scipy.stats import multivariate_normal
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
import functools
import torch
from utils.other_metrics import Segmentation2DMetrics




def _robust_validity_check(*args, checker: Callable[..., bool], **kwargs) -> bool:
    """Wraps ``checker`` function in a try/catch to avoid a crash in the function crashing the whole program.

    Args:
        *args: Additional parameters to pass along to ``checker``.
        checker: Function to wrap in try/catch, whose crash should not interrupt the continuation of the program.
        **kwargs: Additional parameters to pass along to ``checker``.

    Returns:
        Value returned by ``checker`` if call returned, ``False`` if the call to ``checker`` raised an exception.
    """
    try:
        is_valid = checker(*args, **kwargs)
    except Exception:
        is_valid = False
    return is_valid



class RejectionSampler:
    """Generic implementation of the rejection sampling algorithm."""

    def __init__(
        self,
        data: np.ndarray,
        model,
        kde_bandwidth: float = None,
        proposal_distribution_params: Tuple[np.ndarray, np.ndarray] = None,
        scaling_mode: Literal["max", "3rd_quartile"] = "max",
    ):
        """Initializes the inner distributions used by the rejection sampling algorithm.

        Args:
            data: N x D array where N is the number of data points and D is the dimensionality of the data.
            kde_bandwidth: Bandwidth of the kernel density estimator. If no bandwidth is given, it will be determined
                by cross-validation over ``data``.
            proposal_distribution_params: `mean` and `cov` parameters to use for the Gaussian proposal distribution. If
                no params are given, the proposal distribution is inferred from the mean and covariance computed on
                ``data``.
            scaling_mode: Algorithm to use to compute the scaling factor between the proposal distribution and the KDE
                estimation of the real distribution.
        """
        self.data = data
        self.model = model

        # Init kernel density estimate
        if kde_bandwidth:
            self.kde = KernelDensity(bandwidth=kde_bandwidth, kernel="gaussian").fit(self.data)
        else:
            print("Cross-validating bandwidth of kernel density estimate...")
            grid = GridSearchCV(
                KernelDensity(kernel="gaussian"), {"bandwidth": 10 ** np.linspace(-1, 1, 100)}, # cv=ShuffleSplit()
            )
            grid.fit(self.data)
            self.kde = grid.best_estimator_
            print(f"Parameters of KDE optimized for the data: {self.kde}.")
        # bandwidth=0.30538555088334157

        # Init proposal distribution
        if proposal_distribution_params:
            mean, cov = proposal_distribution_params
        else:
            mean = np.mean(self.data, axis=0)               # [ 0.22646421 -0.55352783 -0.64289474  0.27940592  0.1617885   0.09474044]
            cov = np.cov(self.data, rowvar=False)           # [-0.44507677 -0.78859162  0.04096227 ...  1.07716852 -0.631959422.74912487]
            
        self.proposal_distribution = multivariate_normal(mean=mean, cov=cov)

        # Init scaling factor
        factors_between_data_and_proposal_distribution = (
            np.e ** self.kde.score_samples(self.data)               
        ) / self.proposal_distribution.pdf(self.data)               

        # print(self.data, self.data.shape)    # (1272, 32) [ 0.25362736  0.51247966 -0.8736702  ... -1.6859142  -0.56021637 0.00558525]
        
        print("aaaaa", self.proposal_distribution.pdf(self.data).max(),   # 5.258510423008869e-18   
                self.proposal_distribution.pdf(self.data).min(),          # 1.734481550133898e-41
                np.e ** self.kde.score_samples(self.data),                # 4.23904186  0.97127691
                len(np.e ** self.kde.score_samples(self.data)),           # 1272
                factors_between_data_and_proposal_distribution.max(),     # 8.313073320241843e+39 
                factors_between_data_and_proposal_distribution.min())     # 1.675778233949887e+17
        

        if scaling_mode == "max":
            # 'max' scaling mode is used when the initial samples fit in a sensible distribution
            # Should be preferred to other algorithms whenever it is applicable
            self.scaling_factor = np.max(factors_between_data_and_proposal_distribution)
        else:  # scaling_mode == '3rd_quartile'
            # '3rd_quartile' scaling mode is used when outliers in the initial samples skew the ratio and cause an
            # impossibly high scaling factor
            self.scaling_factor = np.percentile(factors_between_data_and_proposal_distribution, 75)

    def sample(self, num_samples: int, batch_size: int = None) -> np.ndarray:
        """Performs rejection sampling to sample M samples that fit the visible distribution of ``data``.

        Args:
            num_samples: Number of samples to sample from the data distribution.
            batch_size: Number of samples to generate in each batch. If ``None``, defaults to ``num_samples / 100``.

        Returns:
            M x D array where M equals `num_samples` and D is the dimensionality of the sampled data.
        """
        # Determine the size and number of batches (possibly including a final irregular batch)
        if batch_size is None:
            batch_size = num_samples // 1000
            print(f"No `batch_size` provided. Defaulted to use a `batch_size` of {batch_size}.")

        batches = [batch_size] * (num_samples // batch_size)   

        if last_batch := num_samples % batch_size:
            batches.append(last_batch)

        # Prepare different seeds for each batch
        ss = SeedSequence()
        print(f"Entropy of root `SeedSequence` used to spawn generators: {ss.entropy}.")
        rngs = [np.random.default_rng(seed) for seed in ss.spawn(len(batches))]     # 100个Generator(PCG64) at 0x1C78000B3C0

        # Sample batches in parallel using a pool of processes          # print(__name__)
        
        if __name__ == 'utils.rejection_sampling':
            with Pool() as pool:
                sampling_result = tqdm(
                    pool.imap(lambda args: self._sample(*args), zip(batches, rngs)),   
                    total=len(batches),
                    desc="Sampling from observed data distribution with rejection sampling",
                    unit="batch",
                )
                # print(sampling_result)
                samples, nb_trials = zip(*sampling_result)         #    , nb_trials.shape
                # print(samples, samples.shape)   
        
        '''
        iterable_value = iter([self._sample(batch) for batch in batches])
        print(iterable_value)
        sampling_result = tqdm( iterable_value,   
                                total=len(batches),
                                desc="Sampling from observed data distribution with rejection sampling",
                                unit="batch",
                                )
        print(sampling_result)
        samples = zip(*sampling_result)     # , nb_trials
        '''

        samples = np.vstack(samples)        # Merge batches of samples in a single array
        nb_trials = sum(nb_trials)          # Sum over the number of points sampled to get each batch

        # Log useful information to analyze/debug the performance of the rejection sampling
        print("nb_trials:", nb_trials)      
        print(f"Number of unique samples generated: {len(np.unique(samples, axis=0))}")
        print(
            "Percentage of generated samples accepted by rejection sampling: "
            f"{round(samples.shape[0] / nb_trials * 100, 2)} \n"
        )

        return samples


    def _sample(self, num_samples: int, rng: np.random.Generator = None) -> Tuple[np.ndarray, int]:
        """Performs rejection sampling to sample M samples that fit the visible distribution of ``data``.

        `self._sample` performs the sampling in itself, as opposed to `self.sample` which is a public wrapper to
        coordinate sampling multiple batches in parallel.

        Args:
            num_samples: Number of samples to sample from the data distribution.
            rng: Random Number Generator to use to draw from both the proposal and uniform distributions.

        Returns:
            samples: M x D array where M equals `num_samples` and D is the dimensionality of the sampled data.
            nb_trials: Number of draws (rejected or accepted) it took to reach M accepted samples. This is mainly useful
                to evaluate the efficiency of the rejection sampling.
        """
        if rng is None:
            rng = np.random.default_rng()

        samples = []
        nb_trials = 0
        while len(samples) < num_samples:

            sample = self.proposal_distribution.rvs(size=1, random_state=rng)   # 从多元正态分布中随机抽取1个样本 (32,) 
            # print("bbbbbbb", sample.max(), sample.min())        # 4.829472361413719 -3.8297241604258154
            # print("ccccccc", self.data.max(), self.data.min())  # 9.735884 -9.290748

            # k = self.proposal_distribution.pdf(sample)          # 4.710602032903247e-24  6.208260603054341e-18
            # print("ddddddd", self.scaling_factor * k.max(), self.scaling_factor * k.min())

            # sample = self.data[0]                               # 4.261145720225105e+16  [4.35322392]
            rand_likelihood_threshold = rng.uniform(0, 10**4 * self.proposal_distribution.pdf(sample))  # self.scaling_factor * 
            # print("eeeeeee", rand_likelihood_threshold)     # 9.971393980192484e-23
            
            # print("fffffff", (np.e ** self.kde.score_samples(sample[np.newaxis, :])))        
            # 1.9803569847963446e+22 [2.02166754e-224] [1.52710164e-08]
            if rand_likelihood_threshold <= (np.e ** self.kde.score_samples(sample[np.newaxis, :])):      
                samples.append(sample)
                # print("ok", len(samples))

            nb_trials += 1

        return np.array(samples), nb_trials



    def _classify_latent_space_anatomical_errors(
        self, latent_space_samples: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Classifies latent space samples based on whether they project to segmentations with anatomical errors or not.

        Args:
            latent_space_samples: N x D array where N is the number of latent space samples and D is the dimensionality
                of the latent space.

        Returns:
            Tuple of two ? x D arrays, where D is the dimensionality of the latent space.
            The first is an M x D array, where M is the number of latent space samples projecting to segmentation maps
            without anatomical errors.
            The second is an K x D array, where K is the number of latent space samples projecting to segmentation maps
            with anatomical errors.
        """
        # A generator that decodes latent space samples by batch (to efficiently process batches at each call)
        # but returns the decoded samples one at a time
        
        def _decoded_latent_samples_generator():

            latent_space_samples_tensor = torch.from_numpy(latent_space_samples)

            self.model.cpu()
            with torch.no_grad():
                decoded_latent_space_samples = [self.model.decode(batch.to(torch.float32)).cpu() for batch in latent_space_samples_tensor] 

            # Yield each decoded sample individually
            for decoded_latent_space_sample in decoded_latent_space_samples:

                yield decoded_latent_space_sample           # torch.Size([1, 1, 608, 832])


        # Wrap anatomical metrics computation in try/catch to avoid crashing the whole process if we fail to compute the
        # anatomical metrics on a particularly degenerated sample that would have been accepted by rejection sampling
        _robust_segmentation_validity_check = functools.partial(
            _robust_validity_check, checker=self.check_segmentation_validity      
        )

        # Process each decoded sample in parallel
        # The result is a nested list of booleans indicating whether the segmentations are anatomically plausible or not
        with Pool() as pool:
            valid_segmentations = list(
                tqdm(
                    pool.imap(_robust_segmentation_validity_check, _decoded_latent_samples_generator()),
                    total=len(latent_space_samples),
                    unit="sample",
                    desc="Classifying latent space samples by the presence of anatomical errors in the projected "
                    "segmentation maps",
                )
            )

        # Convert list to array to be able to invert the values easily
        valid_segmentations = np.array(valid_segmentations)             # [False False False False]
        

        print(
            "Percentage of latent samples that do not produce anatomical errors: "
            f"{sum(valid_segmentations) * 100 / len(latent_space_samples):.2f}"
        )

        return latent_space_samples[valid_segmentations], latent_space_samples[~valid_segmentations]


    def check_segmentation_validity(seg_res):

        seg_res_sq = torch.squeeze(seg_res[:, 0, :, :]) 
        metrics_indicator = Segmentation2DMetrics(seg_res_sq.cpu().data.numpy(), 1)      # 0: background, 1: bone  
        valid_holes = metrics_indicator.count_holes(1)
        valid_connectivity = metrics_indicator.count_disconnectivity(1)     

        if valid_holes!=0 or valid_connectivity!=0:
            return False
        else:
            print("true")
            return True






