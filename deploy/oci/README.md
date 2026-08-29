# Oracle A1 deployment

This is the phase-one, Oracle-only deployment. The full Video Factory, its
browser sessions, SQLite index, render workspace, and publishing backend live
on one Ubuntu VM. There is no public dashboard or job API.

## 1. Create the VM

Create one **Always Free eligible** `VM.Standard.A1.Flex` instance in the OCI
home region with:

- Ubuntu 24.04 ARM64
- 2 OCPUs and 12 GB RAM
- one 200 GB boot volume, avoiding a separate mount and staying within the
  combined 200 GB Always Free block-storage allowance
- SSH access restricted to the administrator's IP whenever possible

Do not expose VNC port 5900 in the OCI security list. The login helper binds it
to localhost and it must be reached through SSH forwarding only.

## 2. Transfer and bootstrap the application

The current working copy is not assumed to have its own public Git remote.
Transfer it from the Mac, excluding local runtime data:

```bash
rsync -az --partial \
  --exclude .git --exclude .venv --exclude workspace --exclude validation \
  --exclude __pycache__ --exclude '*.pyc' --exclude '*.mp4' \
  ./ ubuntu@ORACLE_IP:/tmp/video_factory/
```

This fast-path transfer is only a few megabytes. MoneyPrinterTurbo,
web-scroll-video, Chromium, Python packages, and system packages are downloaded
by the VM over its own network during bootstrap.

On the VM:

```bash
ssh ubuntu@ORACLE_IP
sudo bash /tmp/video_factory/deploy/oci/bootstrap.sh /tmp/video_factory
sudoedit /etc/video-factory/video-factory.env
```

The bootstrap is deliberately pinned to these tested source revisions:

- MoneyPrinterTurbo `d4c0e45da4ac0889af77f7307f52f9d5d4f74942`
- web-scroll-video `7c004aefb8ec4610a18ad21577105a9ddce60b15`
- social-auto-upload is installed by Video Factory at its pinned commit
  `1c66b7db4b30585bbb40c58eb0aa572ffa3cce97`

It installs systemd unit files but does not enable scheduled discovery or
publication.

## 3. Choose a workspace start

For a same-day deployment, start a new canonical workspace on Oracle and copy
only the licensed background track:

```bash
rsync -az --partial \
  workspace/assets/music/858ccdf31193/better-times-are-coming-mixkit-173.mp3 \
  ubuntu@ORACLE_IP:/tmp/better-times-are-coming-mixkit-173.mp3
```

On the VM:

```bash
sudo install -d -o video-factory -g video-factory -m 0750 \
  /srv/video-factory/workspace/assets/music/858ccdf31193
sudo install -o video-factory -g video-factory -m 0640 \
  /tmp/better-times-are-coming-mixkit-173.mp3 \
  /srv/video-factory/workspace/assets/music/858ccdf31193/better-times-are-coming-mixkit-173.mp3
```

The Mac's historical workspace can remain an offline archive and can be
migrated later. This is the recommended path when the goal is to get the PC
offline today.

If historical jobs must be available immediately, stop local generation before
the first full transfer, then run from the Mac:

```bash
rsync -az --partial ./workspace/ \
  ubuntu@ORACLE_IP:/tmp/video-factory-workspace/
```

On the VM:

```bash
sudo rsync -a /tmp/video-factory-workspace/ /srv/video-factory/workspace/
sudo chown -R video-factory:video-factory /srv/video-factory/workspace
```

Do not continue writing to both the Mac and Oracle copies. SQLite and publish
batch state are designed for a single canonical writer.

## 4. Run the ARM and runtime smoke test

```bash
sudo -u video-factory -H \
  /opt/video-factory/app/deploy/oci/smoke-test.sh
```

This verifies ARM64, Node 22+, Deno, FFmpeg/libx264, the Chinese font, the
pinned dependencies, the application CLI, and a headless managed Chromium
launch. Do not enable automation until it passes.

YouTube's pinned PO-token runtime is optional during the first Video Accounts
test. To install it, set `VIDEO_FACTORY_INSTALL_YOUTUBE_RUNTIME=1` in the
environment and rerun bootstrap, or run `video-factory youtube-runtime setup`
as the service user with the production environment loaded.

## 5. Establish the Video Accounts session

Connect from the Mac with VNC forwarding and start the login desktop:

```bash
ssh -L 5901:127.0.0.1:5900 ubuntu@ORACLE_IP
sudo -u video-factory -H \
  /opt/video-factory/app/deploy/oci/login-desktop.sh tencent main
```

While that command is waiting, open `vnc://127.0.0.1:5901` on the Mac and scan
the QR code. Then verify the persisted session:

```bash
sudo -u video-factory -H bash -lc '
  set -a
  source /etc/video-factory/video-factory.env
  set +a
  "$VIDEO_FACTORY_CLI" --workspace "$VIDEO_FACTORY_WORKSPACE" \
    publisher check tencent --account main
'
```

Never expose noVNC or x11vnc directly to the Internet. Browser profiles and
cookies stay under `/srv/video-factory/runtime` and are intentionally excluded
from backup archives.

## 6. Test generation and publication

Run the existing CLI on the VM. Create and approve publish batches there so
their paths remain server-local:

```bash
sudo -u video-factory -H bash -lc '
  set -a
  source /etc/video-factory/video-factory.env
  set +a
  cd "$VIDEO_FACTORY_APP_ROOT"
  "$VIDEO_FACTORY_CLI" --workspace "$VIDEO_FACTORY_WORKSPACE" \
    generate https://github.com/harry0703/MoneyPrinterTurbo
'
```

The existing safety workflow remains unchanged:

```text
publish-create -> human review -> publish-approve -> publish-run
```

Never run `publish-run` for an unreviewed batch. A result marked `uncertain`
must be checked on the platform and must not be retried automatically.

## 7. Enable scheduling and backups

The initial server-safe discovery channels exclude X because the current X
adapter requires OpenCLI attached to a desktop Chrome extension. Edit
`VIDEO_FACTORY_DISCOVERY_CHANNELS` only after adding a server-compatible X
capture backend.

After a manual discovery run succeeds:

```bash
sudo systemctl enable --now video-factory-discovery.timer
sudo systemctl enable --now video-factory-backup.timer
systemctl list-timers 'video-factory-*'
```

Inspect runs with:

```bash
journalctl -u video-factory-discovery.service -n 200 --no-pager
journalctl -u video-factory-backup.service -n 100 --no-pager
```

Local backup archives contain SQLite, manifests, discovery state, and publish
audits. Set `VIDEO_FACTORY_BACKUP_INCLUDE_FINALS=1` to include final MP4 files.
If OCI CLI is separately configured for the `video-factory` user, setting
`OCI_BACKUP_BUCKET` copies the archives to Object Storage.

An archive stored only on the same boot volume is not a disaster-recovery
copy. Configure OCI volume backups or Object Storage before depending on the
Always Free VM; idle Always Free compute can be reclaimed by Oracle.

## Recovery rule

Restore the newest archive into an empty workspace, verify its SHA-256 file,
and rerun `smoke-test.sh`. Platform browser sessions are not restored from
ordinary backups; establish them again through the SSH-only login workflow.
