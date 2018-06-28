# How to use config files

These are TAOS-CI config files for nnstreamer.

- http://suprem.sec.samsung.net/STAR/TAOS-CI

## How to use

After adding TAOS-CI submodule in your project, link them.

```bash
$ mv TAOS-CI/ci/standalone/config/ TAOS-CI/ci/standalone/config.backup
$ ln -s $PWD/Documentation/ci-config $PWD/TAOS-CI/ci/standalone/config
```
