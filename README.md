# Envelope Morphing

[Paper](https://arxiv.org/abs/2506.01588)  

Official code for the WASPAA 2025 paper: Learning Perceptually Relevant Temporal Envelope Morphing. This repository contains code for a framework for learning perceptually meaningful morphs over temporal amplitude envelopes of audio. The code is divided into two main stages:

---

## Stage 1: Envelope Autoencoder Training

First extract amplitude envelopes from audio files and train an autoencoder to learn a compact latent representation. This stage builds on [Stable Audio Tools](https://github.com/Stability-AI/stable-audio-tools).

### Traininng autoencoder:

1. **Install Stable Audio Tools**
   ```bash
   cd stable-audio-tools
   python setup.py install
   ```

2. **Convert Audio to Envelopes**  
   Use the provided utility to extract amplitude envelopes:
   ```bash
   python data_gen.py --input_dir path/to/audio --output_dir path/to/envelopes
   ```

3. **Configure Dataset Paths**  
   Modify `configs/dataset_configs/train.json` and optionally `val.json` to point to your envelope directory.

4. **Train Autoencoder**  
   Make sure your hyperparameters in `defaults.ini` are correct, then run:
   ```bash
   python train.py
   ```

### Using pretrained autoencoder

A pretrained autoencoder checkpoint, trained for 10k steps on the AudioCaps training set, can be downloaded:

```bash
gdown 1n_5b_SXyPkonPb5cc6Sa2NIetunGAB8D
mv autoencoder_ckpt.pth stable-audio-tools/ckpts/
```

---

## Stage 2: Training the Morphing Network

We train a twin (order-invariant) neural network to learn perceptual morphing rules from synthetic envelope pairs. The autoencoder is frozen during this stage.

### Steps:

1. **Generate Synthetic Training Data**
   ```bash
   python data/create_synthetic_data.py
   ```

2. **Train the Morphing Network**
   Adjust hyperparameters as needed, then run:
   ```bash
   python run_training.py
   ```

---

## Citation

If you find this project useful, please cite our paper:

```bibtex
@article{dixit2025learning,
  title={Learning Perceptually Relevant Temporal Envelope Morphing},
  author={Dixit, Satvik and Park, Sungjoon and Donahue, Chris and Heller, Laurie M},
  journal={arXiv preprint arXiv:2506.01588},
  year={2025}
}
```
