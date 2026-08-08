#!/usr/bin/env bash
set -Eeuo pipefail

GROUP=""
ROOT=""

usage() {
  printf 'Usage: %s --group <small|factors|robustness|transfer|imagenet> --root <absolute-path>\n' "$0" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --group)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      GROUP="$2"
      shift 2
      ;;
    --root)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      ROOT="$2"
      shift 2
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

[[ -n "$GROUP" && -n "$ROOT" ]] || { usage; exit 2; }
[[ "$ROOT" = /* ]] || { printf 'error: --root must be absolute\n' >&2; exit 2; }

case "$GROUP" in
  small|factors|robustness|transfer|imagenet) ;;
  *) usage; exit 2 ;;
esac

for tool in curl tar gzip; do
  command -v "$tool" >/dev/null 2>&1 || {
    printf 'error: required command is unavailable: %s\n' "$tool" >&2
    exit 3
  }
done

mkdir -p \
  "$ROOT/archives/$GROUP" \
  "$ROOT/cifar10" \
  "$ROOT/cifar100" \
  "$ROOT/stl10" \
  "$ROOT/factors" \
  "$ROOT/robustness" \
  "$ROOT/cub" \
  "$ROOT/voc" \
  "$ROOT/coco" \
  "$ROOT/imagenet"

ARCHIVES="$ROOT/archives/$GROUP"

fetch() {
  local url="$1"
  local destination="$2"
  local partial="${destination}.part"
  local rc
  mkdir -p "$(dirname "$destination")"
  if [[ -s "$destination" ]]; then
    printf 'SKIP downloaded: %s\n' "$destination"
    return
  fi
  printf 'DOWNLOAD %s\n' "$url"
  set +e
  curl --fail --location --retry 20 --retry-delay 10 --retry-all-errors \
    --connect-timeout 30 --speed-time 120 --speed-limit 1024 \
    --continue-at - --output "$partial" "$url"
  rc=$?
  set -e
  if [[ $rc -eq 33 ]]; then
    printf 'Server cannot resume this partial file; restarting it: %s\n' "$destination"
    curl --fail --location --retry 20 --retry-delay 10 --retry-all-errors \
      --connect-timeout 30 --speed-time 120 --speed-limit 1024 \
      --output "$partial" "$url"
  elif [[ $rc -ne 0 ]]; then
    return "$rc"
  fi
  mv -f "$partial" "$destination"
}

extract_tar() {
  local archive="$1"
  local destination="$2"
  local marker="$3"
  local strip_components="${4:-0}"
  if [[ -f "$marker" ]]; then
    printf 'SKIP extracted: %s\n' "$destination"
    return
  fi
  mkdir -p "$destination"
  if [[ "$strip_components" -gt 0 ]]; then
    tar -xf "$archive" -C "$destination" --strip-components="$strip_components"
  else
    tar -xf "$archive" -C "$destination"
  fi
  touch "$marker"
}

extract_zip() {
  local archive="$1"
  local destination="$2"
  local marker="$3"
  command -v unzip >/dev/null 2>&1 || {
    printf 'error: required command is unavailable: unzip\n' >&2
    exit 3
  }
  if [[ -f "$marker" ]]; then
    printf 'SKIP extracted: %s\n' "$destination"
    return
  fi
  mkdir -p "$destination"
  unzip -oq "$archive" -d "$destination"
  touch "$marker"
}

expand_gzip() {
  local archive="$1"
  local destination="$2"
  local partial="${destination}.part"
  if [[ -s "$destination" ]]; then
    printf 'SKIP expanded: %s\n' "$destination"
    return
  fi
  mkdir -p "$(dirname "$destination")"
  gzip -cd "$archive" > "$partial"
  mv -f "$partial" "$destination"
}

download_small() {
  local archive

  archive="$ARCHIVES/cifar-10-python.tar.gz"
  fetch "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" "$archive"
  extract_tar "$archive" "$ROOT/cifar10" "$ROOT/cifar10/.complete" 1

  archive="$ARCHIVES/cifar-100-python.tar.gz"
  fetch "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz" "$archive"
  extract_tar "$archive" "$ROOT/cifar100" "$ROOT/cifar100/.complete" 1

  archive="$ARCHIVES/stl10_binary.tar.gz"
  fetch "https://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz" "$archive"
  extract_tar "$archive" "$ROOT/stl10" "$ROOT/stl10/.complete" 1

  archive="$ARCHIVES/tiny-imagenet-200.zip"
  fetch "http://cs231n.stanford.edu/tiny-imagenet-200.zip" "$archive"
  if [[ ! -f "$ROOT/tiny-imagenet-200/.complete" ]]; then
    command -v unzip >/dev/null 2>&1 || { printf 'error: unzip is unavailable\n' >&2; exit 3; }
    unzip -oq "$archive" -d "$ROOT"
    touch "$ROOT/tiny-imagenet-200/.complete"
  fi

  fetch "https://raw.githubusercontent.com/google-deepmind/dsprites-dataset/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz" \
    "$ROOT/factors/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
  fetch "https://storage.googleapis.com/3d-shapes/3dshapes.h5" \
    "$ROOT/factors/3dshapes/3dshapes.h5"

  local norb_base="https://cs.nyu.edu/~yann/data/norb-v1.0-small"
  local filename
  for filename in \
    smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz \
    smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz \
    smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz \
    smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz \
    smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz \
    smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz; do
    fetch "$norb_base/$filename" "$ARCHIVES/$filename"
    expand_gzip "$ARCHIVES/$filename" "$ROOT/factors/smallnorb/${filename%.gz}"
  done
}

download_factors() {
  local base="https://huggingface.co/datasets/waleedgondal/mpi3d/resolve/main"
  fetch "$base/mpi3d_toy.npz" "$ROOT/factors/mpi3d/mpi3d_toy.npz"
  fetch "$base/mpi3d_realistic.npz" "$ROOT/factors/mpi3d/mpi3d_realistic.npz"
  fetch "$base/mpi3d_real.npz" "$ROOT/factors/mpi3d/mpi3d_real.npz"
}

download_robustness() {
  local archive
  archive="$ARCHIVES/CIFAR-10-C.tar"
  fetch "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1" "$archive"
  extract_tar "$archive" "$ROOT/robustness/cifar10-c" "$ROOT/robustness/cifar10-c/.complete" 1

  archive="$ARCHIVES/CIFAR-100-C.tar"
  fetch "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1" "$archive"
  extract_tar "$archive" "$ROOT/robustness/cifar100-c" "$ROOT/robustness/cifar100-c/.complete" 1

  local family
  for family in blur digital noise weather extra; do
    archive="$ARCHIVES/imagenet-c-${family}.tar"
    fetch "https://zenodo.org/records/2235448/files/${family}.tar?download=1" "$archive"
    extract_tar "$archive" "$ROOT/robustness/imagenet-c" \
      "$ROOT/robustness/imagenet-c/.${family}.complete"
  done

  archive="$ARCHIVES/imagenet-r.tar"
  fetch "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar" "$archive"
  extract_tar "$archive" "$ROOT/robustness/imagenet-r" "$ROOT/robustness/imagenet-r/.complete" 1

  archive="$ARCHIVES/imagenet-a.tar"
  fetch "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar" "$archive"
  extract_tar "$archive" "$ROOT/robustness/imagenet-a" "$ROOT/robustness/imagenet-a/.complete" 1
}

download_transfer() {
  local archive
  archive="$ARCHIVES/CUB_200_2011.tgz"
  fetch "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1" "$archive"
  extract_tar "$archive" "$ROOT/cub" "$ROOT/cub/.main.complete" 1

  archive="$ARCHIVES/CUB_segmentations.tgz"
  fetch "https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1" "$archive"
  extract_tar "$archive" "$ROOT/cub" "$ROOT/cub/.segmentations.complete"

  archive="$ARCHIVES/VOCtrainval_06-Nov-2007.tar"
  fetch "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar" "$archive"
  extract_tar "$archive" "$ROOT/voc" "$ROOT/voc/.voc2007-trainval.complete" 1

  archive="$ARCHIVES/VOCtest_06-Nov-2007.tar"
  fetch "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar" "$archive"
  extract_tar "$archive" "$ROOT/voc" "$ROOT/voc/.voc2007-test.complete" 1

  archive="$ARCHIVES/VOCtrainval_11-May-2012.tar"
  fetch "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar" "$archive"
  extract_tar "$archive" "$ROOT/voc" "$ROOT/voc/.voc2012-trainval.complete" 1

  archive="$ARCHIVES/coco-train2017.zip"
  fetch "http://images.cocodataset.org/zips/train2017.zip" "$archive"
  extract_zip "$archive" "$ROOT/coco" "$ROOT/coco/.train2017.complete"

  archive="$ARCHIVES/coco-val2017.zip"
  fetch "http://images.cocodataset.org/zips/val2017.zip" "$archive"
  extract_zip "$archive" "$ROOT/coco" "$ROOT/coco/.val2017.complete"

  archive="$ARCHIVES/coco-annotations_trainval2017.zip"
  fetch "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "$archive"
  extract_zip "$archive" "$ROOT/coco" "$ROOT/coco/.instances-annotations.complete"
}

download_imagenet() {
  command -v kaggle >/dev/null 2>&1 || {
    printf 'error: Kaggle CLI is unavailable; official ImageNet access is required\n' >&2
    exit 4
  }
  if [[ ! -f "$HOME/.kaggle/kaggle.json" && -z "${KAGGLE_USERNAME:-}" ]]; then
    printf 'error: Kaggle credentials are unavailable\n' >&2
    exit 4
  fi
  local competition="imagenet-object-localization-challenge"
  local filename
  for filename in \
    ILSVRC2012_img_train.tar \
    ILSVRC2012_img_val.tar \
    ILSVRC2012_devkit_t12.tar.gz; do
    if [[ ! -s "$ARCHIVES/$filename" ]]; then
      kaggle competitions download -c "$competition" -f "$filename" -p "$ARCHIVES"
    fi
  done
  fetch "https://www.image-net.org/data/bboxes_annotations.tar.gz" \
    "$ARCHIVES/bboxes_annotations.tar.gz"
  extract_tar "$ARCHIVES/ILSVRC2012_img_train.tar" "$ROOT/imagenet/train" \
    "$ROOT/imagenet/.train.complete"
  extract_tar "$ARCHIVES/ILSVRC2012_img_val.tar" "$ROOT/imagenet/val" \
    "$ROOT/imagenet/.val.complete"
  extract_tar "$ARCHIVES/ILSVRC2012_devkit_t12.tar.gz" "$ROOT/imagenet" \
    "$ROOT/imagenet/.devkit.complete"
  extract_tar "$ARCHIVES/bboxes_annotations.tar.gz" "$ROOT/imagenet/bboxes" \
    "$ROOT/imagenet/.bboxes.complete"
}

case "$GROUP" in
  small) download_small ;;
  factors) download_factors ;;
  robustness) download_robustness ;;
  transfer) download_transfer ;;
  imagenet) download_imagenet ;;
esac

printf 'COMPLETE group=%s root=%s\n' "$GROUP" "$ROOT"
