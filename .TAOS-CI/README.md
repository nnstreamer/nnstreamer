# How to use config files

These are the TAOS-CI configuration files to support the nnstreamer repository.

- http://github.com/nnsuite/TAOS-CI

## How to use the configruation files
The section describes two steps to use the configuration files of the nnstreamer repository.

#### Step 1
First of all, you must write for passwords carefully as follows.
!!!SECURITY!!! NEVER REVEAL THE BELOW PASSWORDS.

```bash
$ vi ./config/config-environment.sh
TOKEN="xxxxxxxxxxxxxxxxxxxxx"  <---- 1/4: Here!!!
_cov_token="xxxxxxxxxxxxxxxx"  <---- 2/4: Here!!!

$ vi ./config-webhookk.json
{
    "github": {
        "website": "github.com",
        "id": "taos-ci",
        "email": "taos-ci@github.io",
        "secret": "xxxxxxx" <---- 3/4: Here!!!
    },
    "broken_arrow": {
        "id": "admin",
        "pass": "xxxxxxx",  <---- 4/4: Here!!!
        "ip": " " 
    }
}
```

#### Step 2
After adding TAOS-CI submodule in your GitHub project folder, copy and overwrite them.

```bash
$ cp -rf ./conf/* {REPO_DIR}/TAOS-CI/ci/taos/conf/
```
Or you may just link the configuration files with the `ln` command to maintain changes effectively.
```bash
$ cd {REPO_DIR}
$ ln -s .TAOS-CI/conf/config-environment.sh          ./TAOS-CI/ci/taos/conf/config-environment.sh 
$ ln -s .TAOS-CI/conf/config-plugins-postbuild.sh    ./TAOS-CI/ci/taos/conf/config-plugins-postbuild.sh
$ ln -s .TAOS-CI/conf/config-plugins-prebuild.sh     ./TAOS-CI/ci/taos/conf/config-plugins-prebuild.sh
$ ln -s .TAOS-CI/conf/config-server-administrator.sh ./TAOS-CI/ci/taos/conf/config-server-administrator.sh
$ ln -s .TAOS-CI/conf/config-webhook.json            ./TAOS-CI/ci/taos/conf/config-webhook.json
$ ln -s .TAOS-CI/conf/prohibited_words.txt           ./TAOS-CI/ci/taos/conf/prohibited_words.txt

```bash
