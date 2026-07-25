#!/bin/bash
# container_as_ai_mgr.sh — run INSIDE a throwaway render container, as root,
# after pip installs and before dropping privileges. Creates the lab's ai-mgr
# identity (uid 105, host datashare gid 1002) so renders write group-owned,
# group-writable files instead of root-owned ones (owner convention, extended
# to ad-hoc containers 2026-07-24).
#
# Usage in a docker run bash -c string (repo mounted at /sonora):
#   pip install ... ;
#   bash /sonora/scripts/synthesis/container_as_ai_mgr.sh &&
#   runuser -u ai-mgr -- bash -c 'umask 002; python /sonora/scripts/...'
#
# Notes:
# - umask 002 in the runuser shell is part of the contract: setgid dirs give
#   new files the datashare group; 002 makes them group-writable.
# - GPU access: ai-mgr is added to whatever groups own /dev/kfd and the DRI
#   render nodes (host gids appear numerically inside the container).
# - MIOpen find-db: symlinked to the persistent datashare-writable host cache
#   (gfx1151 cold-db ≈1h fake-hang trap).
set -u
groupadd -g 1002 datashare 2>/dev/null
getent group 109 >/dev/null || groupadd -g 109 ai-mgr
useradd -u 105 -g 109 -G 1002 -m -s /bin/bash ai-mgr 2>/dev/null
for d in /dev/kfd /dev/dri/renderD*; do
  [ -e "$d" ] || continue
  gid=$(stat -c %g "$d")
  getent group "$gid" >/dev/null || groupadd -g "$gid" "hw$gid"
  usermod -aG "$(getent group "$gid" | cut -d: -f1)" ai-mgr
done
mkdir -p /home/ai-mgr/.config
# MIOpen's user find-db must be OWNED by the running user: it calls
# std::filesystem::permissions() (chmod) on the .ufdb file, and chmod checks
# ownership, not write bits. Pointing ai-mgr at the host-user-owned cache made
# every render abort with "cannot set permissions: Operation not permitted"
# mid-bank, even at mode 777 (owner finding 2026-07-25 — it killed 3 of 4 engines
# in the teacher-ab-v1 run). ai-mgr therefore gets its OWN cache dir, seeded from
# the shared one so it is not cold (the gfx1151 cold-db ≈1h fake-hang trap).
ln -sfn /data/model-training/sonora/miopen-ai-mgr /home/ai-mgr/.config/miopen
chown -R 105:109 /home/ai-mgr 2>/dev/null
echo "ai-mgr ready: $(id ai-mgr)"
