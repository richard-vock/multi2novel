# Multi-View to Novel View

This is a pytorch implementation of the paper "Multi-view to Novel view: Synthesizing Novel Views with Self-Learned Confidence" by Sun et al.

- Project homepage: https://shaohua0116.github.io/Multiview2Novelview/
- Paper: https://shaohua0116.github.io/Multiview2Novelview/sun2018multiview.pdf
- Appendix: https://shaohua0116.github.io/Multiview2Novelview/sun2018multiview_supp.pdf
- Original TensorFlow implementation: https://github.com/shaohua0116/Multiview2Novelview

This implementation is based on the author's TensorFlow implementation but diverges in some aspects (where it is closer to description in the published paper - see *Implementation Details* below).

Tensorboard visualization and Multi-GPU support are included.

## Running the Code

This implementation was tested in the `vastai/pytorch` docker hub image on the KITTI dataset. Dependencies stated are therfore relative to this docker image (YMMV).

### Dependencies

The only strict dependency is `h5py`. Optional dependencies (in the sense that you can remove corresponding code easily without breaking core functionality) are `tensorboard` (for monitoring/visualizing training) and `tqdm` (for progress bar output).

For the above mentioned docker container a

    pip install --upgrade torch tensorboard tqdm h5py

should suffice.

### Datasets

- Download one of the original datasets (you can find the dataset in the TF implementation readme: https://github.com/shaohua0116/Multiview2Novelview)
  This implementation assumes the files:
  - `/path/to/dataset/data.hdf5`
  - `/path/to/dataset/id_train.txt`
  - `/path/to/dataset/id_test.txt`

### Execution

Assuming the KITTI dataset at `/data/kitti` you can start training by executing

    ./train.py --root /data/kitti

 Checkpoints are written to `./checkpoints`. For additional parameters please consult the top of the main function in `train.py`/`infer.py`.


## Implementation Details

The original TensorFlow implementation differs from the description in the paper. This mostly concerns the aggregation of pixel prediction / flow estimation output.
In most cases this implementation favors the version of the paper - details where the TF variants were used are:

- The choice of an L1-norm instead of L2 for the confidence loss.
- Choices for weights and arbitrary cost factors in the loss function.
- Using different optimizer parameter sets for flow estimation, pixel prediction and discriminator.

## Performance

Since the paper does not state any timings here are my results on a single GeForce 2080Ti:

- Forward pass (pure, i.e. without gradient computation): 230ms
- Average training time per batch (forward+backward+loss/optimizer step): 416ms
- One training epoch on KITTI dataset, 2320 batches of 8 samples: 16m04s
