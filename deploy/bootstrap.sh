#!/usr/bin/env bash
#
# One-time hardening for a fresh Ubuntu 24.04 VM. Run as root, once, before
# anything else touches the box:
#
#     ssh root@<ip> 'bash -s' < deploy/bootstrap.sh
#
# Idempotent — safe to re-run after changing something.
#
# What this does, in the order that matters:
#   1. patches, and keeps patching itself
#   2. creates the deploy user (you will not run the app as root)
#   3. locks SSH down to key-only, no root
#   4. firewall: 22/80/443 in, everything else dropped
#   5. fail2ban on SSH
#   6. Docker from Docker's own repo, not Ubuntu's ancient one
#
# THE SSH STEP CAN LOCK YOU OUT, so it refuses to run unless an authorized
# key is already installed for the deploy user. That check is the whole
# reason this is a script and not a list of commands in a README — the
# failure mode of "disable password auth, then discover the key never
# copied" is a rebuilt server.

set -euo pipefail

DEPLOY_USER="${DEPLOY_USER:-deploy}"
SSH_PORT="${SSH_PORT:-22}"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "run as root"

# ---------------------------------------------------------------------------
log "1/6  Patching, and enabling unattended security upgrades"
# Most compromises of a small VM are not clever. They are an unpatched
# service and a scanner. Automatic security updates close more real risk
# than any other line in this file.
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get upgrade -y -qq
apt-get install -y -qq ca-certificates curl gnupg ufw fail2ban \
    unattended-upgrades apt-listchanges

cat >/etc/apt/apt.conf.d/20auto-upgrades <<'EOF'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
APT::Periodic::AutocleanInterval "7";
EOF

# Reboot at 04:00 when a patch needs it. An unpatched kernel that is waiting
# politely for a manual reboot is an unpatched kernel.
sed -i 's|^//\s*Unattended-Upgrade::Automatic-Reboot ".*";|Unattended-Upgrade::Automatic-Reboot "true";|' \
    /etc/apt/apt.conf.d/50unattended-upgrades
sed -i 's|^//\s*Unattended-Upgrade::Automatic-Reboot-Time ".*";|Unattended-Upgrade::Automatic-Reboot-Time "04:00";|' \
    /etc/apt/apt.conf.d/50unattended-upgrades

# ---------------------------------------------------------------------------
log "2/6  Creating the '${DEPLOY_USER}' user"
if ! id -u "$DEPLOY_USER" >/dev/null 2>&1; then
    adduser --disabled-password --gecos "" "$DEPLOY_USER"
fi
usermod -aG sudo "$DEPLOY_USER"

# Carry root's authorized keys over, so the key you already used to get here
# keeps working as the deploy user.
install -d -m 700 -o "$DEPLOY_USER" -g "$DEPLOY_USER" "/home/$DEPLOY_USER/.ssh"
if [ -f /root/.ssh/authorized_keys ] && [ ! -s "/home/$DEPLOY_USER/.ssh/authorized_keys" ]; then
    cp /root/.ssh/authorized_keys "/home/$DEPLOY_USER/.ssh/authorized_keys"
    chown "$DEPLOY_USER:$DEPLOY_USER" "/home/$DEPLOY_USER/.ssh/authorized_keys"
    chmod 600 "/home/$DEPLOY_USER/.ssh/authorized_keys"
fi

# ---------------------------------------------------------------------------
log "3/6  Locking down SSH"
# The check that stops this being a foot-gun.
if [ ! -s "/home/$DEPLOY_USER/.ssh/authorized_keys" ]; then
    die "no authorized_keys for ${DEPLOY_USER}.
     Disabling password auth now would lock you out permanently.
     From your laptop:  ssh-copy-id ${DEPLOY_USER}@<this-host>
     then re-run this script."
fi

cat >/etc/ssh/sshd_config.d/99-hardening.conf <<EOF
Port ${SSH_PORT}
PermitRootLogin no
PasswordAuthentication no
KbdInteractiveAuthentication no
ChallengeResponseAuthentication no
PubkeyAuthentication yes
PermitEmptyPasswords no
X11Forwarding no
AllowAgentForwarding no
MaxAuthTries 3
LoginGraceTime 20
AllowUsers ${DEPLOY_USER}
EOF

sshd -t || die "sshd config invalid — NOT restarting. Fix before disconnecting."
systemctl reload ssh || systemctl reload sshd

# ---------------------------------------------------------------------------
log "4/6  Firewall"
# Default deny inbound. Postgres, gunicorn and node are reachable only
# inside the compose network, so they need no rule here — and must never
# get one.
ufw --force reset >/dev/null
ufw default deny incoming
ufw default allow outgoing
ufw allow "${SSH_PORT}/tcp" comment 'ssh'
ufw allow 80/tcp  comment 'http (acme + redirect)'
ufw allow 443/tcp comment 'https'
ufw --force enable

# ---------------------------------------------------------------------------
log "5/6  fail2ban on SSH"
cat >/etc/fail2ban/jail.d/sshd.local <<EOF
[sshd]
enabled  = true
port     = ${SSH_PORT}
backend  = systemd
maxretry = 4
findtime = 10m
bantime  = 1h
EOF
systemctl enable --now fail2ban
systemctl restart fail2ban

# ---------------------------------------------------------------------------
log "6/6  Docker (from Docker's repo — Ubuntu's package is years behind)"
if ! command -v docker >/dev/null 2>&1; then
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        >/etc/apt/sources.list.d/docker.list
    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io \
        docker-buildx-plugin docker-compose-plugin
fi
usermod -aG docker "$DEPLOY_USER"
systemctl enable --now docker

# A second line of defence on log growth, for any container that somehow
# starts without the limits in docker-compose.prod.yml.
cat >/etc/docker/daemon.json <<'EOF'
{
  "log-driver": "json-file",
  "log-opts": { "max-size": "10m", "max-file": "5" },
  "live-restore": true
}
EOF
systemctl restart docker

# ---------------------------------------------------------------------------
log "Done."
cat <<EOF

  Verify BEFORE you close this session — if SSH is broken, this shell is the
  only way back in:

      ssh ${DEPLOY_USER}@<this-host> 'docker ps && sudo ufw status'

  Then, as ${DEPLOY_USER}:
      git clone <repo> ~/cubearena && cd ~/cubearena
      cp deploy/.env.prod.example .env && chmod 600 .env && \$EDITOR .env
      ./deploy/deploy.sh

EOF
