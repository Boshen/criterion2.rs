#!/usr/bin/env -S just --justfile

_default:
  @just --list -u

watch command:
  cargo watch -x '{{command}}'
