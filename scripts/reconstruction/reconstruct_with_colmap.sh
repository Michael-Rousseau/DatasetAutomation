#!/bin/zsh
# Sparse reconstruction of a set of images with COLMAP, matching only the pairs already known.
#
#   scripts/reconstruction/reconstruct_with_colmap.sh <image folder> <workspace> [kept_images.txt]
#
# The workspace receives the COLMAP database, the model (sparse/, and sparse_text/ as text
# with points.ply) and one log per stage; the time of each stage is printed. Without a list of
# kept images, every image of the latest sequential run is reconstructed.
#
# Runs on the CPU: about 7 minutes for 600 images of 3840 x 2880 on 11 cores. The dense
# reconstruction is not attempted, COLMAP needs CUDA for it.
#
# Choices: one camera for all the images, with distortion (OPENCV model), calibrated by COLMAP
# itself; features on images reduced to 1920 pixels; the global mapper, three times faster
# than the incremental one on 150 images for the same camera positions.
set -eu
IMAGES=$1
W=$2
KEPT=${3:-}

stage() {
  local name=$1; shift
  local start=$SECONDS
  "$@" > $W/log_$name.txt 2>&1
  echo "$name: $((SECONDS - start)) s"
}

mkdir -p $W
rm -rf $W/database.db $W/database.db-shm $W/database.db-wal $W/sparse $W/sparse_text
mkdir -p $W/sparse $W/sparse_text
# An array: zsh would pass "--kept <file>" written inline as a single word.
selection=()
[[ -n $KEPT ]] && selection=(--kept $KEPT)
uv run python scripts/reconstruction/export_colmap_pairs.py $IMAGES --output $W $selection
stage extract colmap feature_extractor --database_path $W/database.db --image_path $IMAGES \
  --image_list_path $W/image_list.txt --ImageReader.single_camera 1 \
  --ImageReader.camera_model OPENCV --FeatureExtraction.use_gpu 0 \
  --FeatureExtraction.max_image_size 1920
stage match colmap matches_importer --database_path $W/database.db \
  --match_list_path $W/pairs.txt --match_type pairs --FeatureMatching.use_gpu 0
stage map colmap global_mapper --database_path $W/database.db --image_path $IMAGES \
  --output_path $W/sparse
for model in $W/sparse/*(/N); do
  text=$W/sparse_text/$(basename $model)
  mkdir -p $text
  colmap model_converter --input_path $model --output_path $text --output_type TXT > /dev/null 2>&1
  colmap model_converter --input_path $model --output_path $text/points.ply --output_type PLY > /dev/null 2>&1
  echo "model $(basename $model):"
  colmap model_analyzer --path $model 2>&1 | grep -E "Registered images|Points|Mean" | sed 's/.*\] //'
done
