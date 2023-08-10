#!/usr/bin/env bash

mkdir -p release

export MACOSX_DEPLOYMENT_TARGET=10.14

for TARGET in aarch64-apple-darwin x86_64-apple-darwin; do
  CARGO_PROFILE_RELEASE_LTO=fat PYO3_NO_PYTHON=1 maturin build --release --strip --target $TARGET
done

cp target/wheels/*.whl release/
