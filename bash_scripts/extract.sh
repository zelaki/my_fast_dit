# torchrun --nnodes=1 --nproc_per_node=8  extract_features.py --data-path /data/imagenet/train --features-path /data/imagenet/feats/ 
# torchrun --nnodes=1 --nproc_per_node=8  extract_features.py --data-path /data/imagenet/train --features-path /data/imagenet/feats-f8-kl/  --vae ours --vae-ckpt /home/ubuntu/latent-diffusion/kl-f8/model.ckpt
# torchrun --nnodes=1 --nproc_per_node=8  extract_features.py --data-path /data/imagenet/train --features-path /data/imagenet/feats-f8-kl-scale-equi-uniform  --vae ours --vae-ckpt /home/ubuntu/latent-diffusion/logs/rresume-kl-f8-openimages-scale-equi-uniform/checkpoints/save_epoch_5.ckpt  



# torchrun --nnodes=1 --nproc_per_node=8  extract_features.py --data-path /data/imagenet/train --features-path /data/imagenet/feats-f8-kl-ft-kl=0.0005  --vae ours --vae-ckpt /home/ubuntu/latent-diffusion/logs/resume-kl-f8-openimages-ft-kl-0.00005/checkpoints/epoch=000005.ckpt  

# torchrun --nnodes=1 --nproc_per_node=8  extract_features.py --data-path /data/imagenet/train --features-path /data/imagenet/feats-f8-kl-equi-prior-preserve  --vae ours --vae-ckpt /home/ubuntu/latent-diffusion/logs/resume-kl-f8-openimages-equi-prior_preserve/checkpoints/save_epoch_5.ckpt


# torchrun --nnodes=1 --nproc_per_node=8  extract_features.py --data-path /data/imagenet/train --features-path /data/imagenet/feats-f8-kl-equi-unifrom-sampling-lb-0.5  --vae ours --vae-ckpt /home/ubuntu/latent-diffusion/logs/resume-kl-f8-openimages-equi-unifrom-sampling-lb-0.5/checkpoints/save_epoch_5.ckpt

# torchrun --nnodes=1 --nproc_per_node=8  extract_features.py --data-path /data/imagenet/train --features-path equi-scale-flip-prior-preserve --vae ours --vae-ckpt /home/ubuntu/latent-diffusion/logs/resume-kl-f8-openimages-equi-scale-flip-prior-preserve/checkpoints/save_epoch_5.ckpt

# torchrun --nnodes=1 --nproc_per_node=8  extract_features.py --data-path /data/imagenet/train --features-path /data/imagenet/feats-f8-kl-equi-equi-anisotropic-scale-uniform-0.5-flip-prior-preserve  --vae ours --vae-ckpt ~/latent-diffusion/logs/resume-kl-f8-openimages-equi-anisotropic-scale-uniform-0.5-flip-prior-preserve/checkpoints/save_epoch_5.ckpt

# torchrun --nnodes=1 --nproc_per_node=8  extract_features.py --data-path /data/imagenet/train --features-path /data/imagenet/feats-vae-sd-xl-scale-flip-prior-preserve  --vae ours --vae-ckpt /data/f8-kl-ckpt/resume-xl-openimages-scale-flip/ckeckpoints/epoch=000005.ckpt


# torchrun --nnodes=1 --nproc_per_node=8  extract_features.py --data-path /data/imagenet/train --features-path /data/imagenet/feats-vae-sd3 --vae-scaling-factor 1.5305  --vae sd3


torchrun --nnodes=1 --nproc_per_node=8  extract_features.py --data-path /data/imagenet/train --features-path /data/imagenet/feats-vae-sd3-scale-flip-prior-preserve  --vae ours --vae-type sd3 --vae-ckpt /home/ubuntu/latent-diffusion/logs/resume-kl-f8-openimages-sd3-scale-flip/2025-01-07T18-00-39_autoencoder_kl_wrapper_sd3_32x32x4_openimages/checkpoints/save_epoch_5.ckpt --vae-config /home/ubuntu/latent-diffusion/configs/autoencoder/autoencoder_kl_wrapper_sd3_32x32x4_openimages.yaml --vae-scaling-factor 1.5305







