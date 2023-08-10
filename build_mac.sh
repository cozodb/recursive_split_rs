#!/usr/bin/env bash

CARGO_PROFILE_RELEASE_LTO=fat PYO3_NO_PYTHON=1 maturin build --release --strip