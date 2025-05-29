# Part 2 of hackathon submission (17/12/2024) for tip deconvolution.

**Model 2:** Autoencoder with 33k parameters. It was trained for 1000 epochs on the dataset of 1200 STM images (128x128 px) with normalization. Loss was MAE.​

**Model 3:** Autoencoder with 6.7 million parameters and skip connections. It was trained for 75 epochs on the dataset of 100 STM images (256 x256 px) with normalization. Loss was MAE combined with SSIM as well as a custom gradient loss to help reduce blur in reconstructed image and add more detail. Ideally, this model should be run for more epochs but due to time constraints was limited to only 75 epochs. The reconstructed image shows promise for getting rid of blunt tip affect but is blurry in regions lacking much detail. Further hyperparameter tuning, lower learning rate and optimized loss function weights(more SSIM and gradient loss)  could possibly help gain clearer reconstruction image. ​

**Link to the repository with first model:**
https://github.com/nickkolev97/Tip_deconvolution_hack

**Participants:**
Nick, Mikhail Petrov, Viktoriia Liu, Sergey Ilyev, Srikar Rairao, Tommaso Rodani
