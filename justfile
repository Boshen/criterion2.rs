#!/usr/bin/env -S just --justfile

_default:
  @just --list -u

alias r := ready

init:
  cargo binstall cargo-watch typos-cli taplo-cli -y

ready:
  git diff --exit-code --quiet
  typos
  just fmt
  just check
  cargo test

watch command:
  cargo watch -x '{{command}}'

check:
  cargo check --all-features

fmt:
  cargo fmt
  taplo format
