# Wasserstein-DCGAN
Wasserstein DCGAN code based on carpedm20/DCGAN-tensorflow

1) download carpedm20/DCGAN-tensorflow code
2) download wmodel.py
3) edit main.py, change: from wmodel import DCGAN
4) run with same parameters

just implement Wasserstein GAN algorithm base on original DCGAN code structure,
if you've read the origal DCGAN code, you'd find the changed spots easily:
the 2 new loss functions, clipping operation, different optimization method, discards sigmoid from discriminator output.


some observations:
1) in utils.visualize(), option=1, 
    use z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
    instead of nearly all zero sparse values, would generate more variant outputs (same digit looks much different)
2) in model.py train()
    the sample noise input:
    sample_z = np.random.uniform(-0.5, 0.5, size=(self.sample_num , self.z_dim))
    use 0.5 instead of 1, the sample outputs would look better, it seems the smaller noise abs values the better output quality, but less variation.
